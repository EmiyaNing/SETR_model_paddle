import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from models import Transformer
from dataload import Dataloader, Transform
from Data_Augement import Augmentation
from Lossfunc import CostFunc

parse = argparse.ArgumentParser()
parse.add_argument('--image_size', type=int, default=480)
parse.add_argument('--num_class', type=int, default=16)
parse.add_argument('--hidden_unit_num', type=int, default=1024)
parse.add_argument('--layer_num', type=int, default=20)
parse.add_argument('--head_num', type=int, default=8)
parse.add_argument('--dropout', type=float, default=0.8)
parse.add_argument('--decoder_name', type=str, default='PUP')
parse.add_argument('--lr_init', type=float, default=0.01)
parse.add_argument('--num_epochs', type=int, default=1000)
parse.add_argument('--batch_size', type=int, default=8)
parse.add_argument('--image_folder', type=str, default='/home/aistudio/dataset')
parse.add_argument('--image_list_file', type=str, default='/home/aistudio/dataset/train_list.txt')
parse.add_argument('--val_list_file', type=str, default='/home/aistudio/dataset/val_list.txt')
parse.add_argument('--checkpoint_folder', type=str, default='./output')
parse.add_argument('--save_freq', type=int, default=20)

args = parse.parse_args()

def combine_channels(input):
    new_data = np.zeros([input.shape[0], 1, input.shape[2], input.shape[3]], dtype='int64')
    np_data  = input.numpy()
    for i in range(args.num_class):
        sub_matrix = np_data[:, i, :, :]
        sub_matrix = sub_matrix[:, np.newaxis, :, :]
        sub_matrix[sub_matrix >  0.5] = i
        sub_matrix[sub_matrix <= 0.5] = 0
        new_data[sub_matrix == i] = i
    return new_data

def train(train_load, model, costFunc, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    miou_meter       = AverageMeter()
    for batch_id, data in enumerate(train_load):
        image = data[0].astype('float32')
        label = data[1]
        image = fluid.layers.transpose(image, perm=(0, 3, 1, 2))

        preds = model(image)
        loss  = costFunc(preds, label)
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()
        n = image.shape[0]
        preds  = fluid.layers.softmax(preds, axis=1)
        pred_label = fluid.layers.argmax(preds, axis=1)
        miou,_,_  = fluid.layers.mean_iou(pred_label, label, args.num_class)
        train_loss_meter.update(loss.numpy()[0], n)
        miou_meter.update(miou.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Average Loss: {train_loss_meter.avg:4f}, "+
                f"Mean Iou: {miou_meter.avg:4f}") 
    return train_loss_meter.avg

'''def validation(dataloader, val_size, model, num_classes):
    # TODO: validation phase.
    model.eval()
    accuracies = []
    mious = []
    counter = 0
    for image,label in dataloader():
        counter += 1

        image = fluid.layers.transpose(image, perm=[0, 3, 1, 2])
        pred = model(image)
        pred = fluid.layers.softmax(pred, axis=1)
        pred_label = fluid.layers.argmax(pred,axis=1)
        # NCHW -> NHWC
        pred = fluid.layers.transpose(pred,perm=[0, 2, 3, 1])


        miou, _, _ = paddle.fluid.layers.mean_iou(pred_label, label, num_classes)
        mious.append(miou.numpy())
        print(miou.numpy()[0])'''


def main():
    Place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(Place):
        #preprocess = Transform(480)
        preprocess = Augmentation(480)
        dataloader = Dataloader(args.image_folder, args.image_list_file, transform=preprocess, shuffle=True)
        dataloader_1 = Dataloader(args.image_folder, args.val_list_file)
        train_load = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        train_load.set_sample_generator(dataloader, batch_size=args.batch_size, places=Place)
        #val_load   = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        #val_load.set_sample_generator(dataloader, batch_size=args.batch_size, places=Place)
        total_batch = int(len(dataloader) / args.batch_size)

        model = Transformer(args.image_size, args.num_class, args.hidden_unit_num, args.layer_num, args.head_num, args.dropout, args.decoder_name, False)

        costFunc = CostFunc
        optimizer= SGDOptimizer(fluid.layers.polynomial_decay(args.lr_init, 100, power=0.9), parameter_list=model.parameters())

        for epoch in range(1, args.num_epochs + 1):
            train_loss = train(train_load, model, costFunc, optimizer, epoch, total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"SETR-Epoch-{epoch}-Loss-{train_loss}-Miou-{miou}")

                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')
            #validation(val_load, val_size = 5, model = model, num_classes=args.num_class)   
if __name__ == "__main__":
    main()