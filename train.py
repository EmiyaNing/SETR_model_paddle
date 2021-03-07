import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import SGDOptimizer
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from models import Transformer
from dataload import Dataloader, Transform
from Data_Augement import Augmentation
from Lossfunc import CostFunc

parse = argparse.ArgumentParser()
parse.add_argument('--image_size', type=int, default=512)
parse.add_argument('--num_class', type=int, default=15)
parse.add_argument('--hidden_unit_num', type=int, default=512)
parse.add_argument('--layer_num', type=int, default=2)
parse.add_argument('--head_num', type=int, default=16)
parse.add_argument('--dropout', type=float, default=0.8)
parse.add_argument('--decoder_name', type=str, default='PUP')
parse.add_argument('--lr_init', type=float, default=0.01)
parse.add_argument('--num_epochs', type=int, default=10)
parse.add_argument('--batch_size', type=int, default=1)
parse.add_argument('--image_folder', type=str, default='/home/aistudio/dataset')
parse.add_argument('--image_list_file', type=str, default='/home/aistudio/dataset/train_list.txt')
parse.add_argument('--val_list_file', type=str, default='/home/aistudio/dataset/val_list.txt')
parse.add_argument('--checkpoint_folder', type=str, default='./output')
parse.add_argument('--save_freq', type=int, default=1)

args = parse.parse_args()


def train(train_load, model, costFunc, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    miou_meter       = AverageMeter()
    count            = 0
    for batch_id, data in enumerate(train_load):
        image = data[0].astype('float32')
        label = data[1]
        image = fluid.layers.transpose(image, perm=(0, 3, 1, 2))

        preds, auxiliary = model(image)
        loss  = costFunc(preds, label)
        beta  = costFunc(auxiliary, label)
        loss  = loss + beta * 0.5 
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()
        n = image.shape[0]
        preds  = fluid.layers.softmax(preds, axis=1)
        pred_label = fluid.layers.argmax(preds, axis=1)
        miou,_,_  = fluid.layers.mean_iou(pred_label, label, args.num_class)
        train_loss_meter.update(loss.numpy()[0], n)
        miou_meter.update(miou.numpy()[0], n)
        if count == 8:
            print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                    f"Step[{batch_id:04d}/{total_batch:04d}], " +
                    f"Average Loss: {train_loss_meter.avg:4f}, "+
                    f"Mean Iou: {miou_meter.avg:4f}") 
            count = 0
        else:
            count += 1
    return train_loss_meter.avg

def validation(dataloader, val_size, model, num_classes):
    # TODO: validation phase.
    model.eval()
    accuracies = []
    mious = AverageMeter()
    counter = 0
    for image,label in dataloader():
        counter += 1

        image = fluid.layers.transpose(image, perm=[0, 3, 1, 2])
        pred,aux = model(image)
        pred = fluid.layers.softmax(pred, axis=1)
        pred_label = fluid.layers.argmax(pred,axis=1)
        # NCHW -> NHWC
        pred = fluid.layers.transpose(pred,perm=[0, 2, 3, 1])


        miou, _, _ = paddle.fluid.layers.mean_iou(pred_label, label, num_classes)
        mious.update(miou.numpy()[0], 1)
        if counter == val_size:
            return mious.avg


def main():
    Place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(Place):
        #preprocess = Augmentation(args.image_size)
        preprocess = Transform(args.image_size)
        dataloader = Dataloader(args.image_folder, args.image_list_file, transform=preprocess, shuffle=True)
        dataloader_1 = Dataloader(args.image_folder, args.val_list_file, transform=preprocess, shuffle=True)
        train_load = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        train_load.set_sample_generator(dataloader, batch_size=args.batch_size, places=Place)
        val_load   = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        val_load.set_sample_generator(dataloader_1, batch_size=args.batch_size, places=Place)
        total_batch = int(len(dataloader) / args.batch_size)

        model = Transformer(image_size=args.image_size,
                            num_classes=args.num_class,
                            hidden_unit_num=args.hidden_unit_num,
                            layer_num=args.layer_num,
                            head_num=args.head_num,
                            dropout=args.dropout,
                            decoder_name='PUP',
                            hyber=True,
                            visualable=False)

        costFunc = CostFunc
        #optimizer= SGDOptimizer(fluid.layers.polynomial_decay(args.lr_init, 10, power=0.9), parameter_list=model.parameters())
        optimizer = AdamOptimizer(fluid.layers.polynomial_decay(args.lr_init, 10, power=0.9), parameter_list=model.parameters())
        for epoch in range(1, args.num_epochs + 1):
            train_loss = train(train_load, model, costFunc, optimizer, epoch, total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss}")

            miou = validation(val_load, val_size = 256, model = model, num_classes=args.num_class)   
            print("------Now the Mean IOU == " + str(miou) + "------------")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"SETR-ResNet30-Epoch-{epoch}-Loss-{train_loss:4f}-MIOU-{miou:4f}")

                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')


if __name__ == "__main__":
    main()