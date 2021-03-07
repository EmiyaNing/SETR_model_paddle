import cv2
import os
import paddle
import numpy as np
import paddle.fluid as fluid
from imghdr import what
from paddle.fluid.dygraph import load_dygraph
from dataload import Dataloader,Transform
from models import Transformer
from utils import AverageMeter

save_path = "/home/aistudio/predicts/"
val_path  = "/home/aistudio/dataset/"

class Load_infer():
    def __init__(self,image_path, file_list, transform = None, shuffle = False):
        self.image_path = image_path
        self.file_list  = file_list
        self.transform  = transform
        self.shuffle    = shuffle


    def pre_process(self, data):
        output_data =  cv2.resize(data, (512, 512), interpolation=cv2.INTER_LINEAR).astype('float32') / 255
        return output_data

    def __len__(self):
        return len(self.file_list)

    def __call__(self):
        for data_path in self.file_list:
            input_data = cv2.imread(data_path)
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)      
            input_data = self.pre_process(input_data)
            label      = np.random.rand(input_data.shape[0], input_data.shape[1], input_data.shape[2])
            yield input_data, label


def get_infer_data(dir_images):
    res = []
    for file in os.listdir(dir_images):
        image_path = os.path.join(dir_images, file)
        if os.path.exists(image_path):  
            if what(image_path):
                res.append(os.path.join(dir_images, file))
        else:
            print(image_path)
    return res


# process_image 可能有非常严重的bug
'''def process_image(model, path_list):
    process  = Transform(512)
    for id_num,path in enumerate(path_list):
        name  = path.split("/")
        file_name = name[len(name)-1]
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        high, width, depth = image.shape
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR).astype('float32') / 255
        data  = image[np.newaxis, :, :, :]
        data  = fluid.dygraph.to_variable(data)
        data  = fluid.layers.transpose(data, perm=(0, 3, 1, 2))
        pred, auxiliary = model(data)
        pred  = fluid.layers.softmax(pred)
        pred_label      = fluid.layers.argmax(pred, axis=1)
        pred_label      = fluid.layers.reshape(pred_label, [pred_label.shape[1], pred_label.shape[2]])
        res_image       = pred_label.numpy().astype('uint8')
        res_image       = cv2.resize(res_image, (high, width), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(save_path + file_name, res_image)
        print(res_image)
        if id_num % 10 == 0:
            print("---------------Now have process image " + str(id_num) + "----------")'''

def value_check(image):
    high, width = image.shape
    flag = True
    for i in range(high):
        for j in range(width):
            if image[i][j] != 0:
                flag = False
                break
    return flag
            

def validation(dataloader, model, num_classes):
    # TODO: validation phase.
    accuracies = []
    mious = AverageMeter()
    counter = 0
    for image,label in dataloader():
        counter += 1

        image = fluid.layers.transpose(image, perm=[0, 3, 1, 2])
        pred,aux = model(image)
        pred = fluid.layers.softmax(pred, axis=1)
        pred_label = fluid.layers.argmax(pred,axis=1)
        res_file   = fluid.layers.reshape(pred_label, [pred_label.shape[1], pred_label.shape[2]])
        res_file   = res_file.numpy()
        flag       = value_check(res_file)
        if flag:
            print("empty image")
        else:
            print("not empty")
            miou, _, _ = paddle.fluid.layers.mean_iou(pred_label, label, num_classes)
            mious.update(miou.numpy()[0], 1)
            if counter %10 == 0:
                print(mious.avg)

def process_image(model, dataloader, file_list):
    count = 0
    for image,label in dataloader:

        image = fluid.layers.transpose(image, perm=[0, 3, 1, 2])
        pred,aux  = model(image)
        pred = fluid.layers.softmax(pred, axis=1)
        pred_label = fluid.layers.argmax(pred,axis=1)
        pred_label = fluid.layers.reshape(pred_label, [image.shape[2], image.shape[3]])
        res_file   = pred_label.numpy().astype('uint8')
        res_file   = cv2.resize(res_file, (1536, 1536), interpolation=cv2.INTER_NEAREST)
        name  = file_list[count].split("/")
        file_name = name[len(name)-1]
        cv2.imwrite(save_path + file_name, res_file)
        count += 1
        if count % 10 ==0:
            print("Now have process image " + str(count) + ".....")




def main():
    Place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(Place):
        model = Transformer(image_size=512,
                            num_classes=15,
                            hidden_unit_num=1024,
                            layer_num=2,
                            head_num=16,
                            dropout=0.8,
                            decoder_name='PUP',
                            hyber=True,
                            visualable=False)
        preprocess = Transform(512)
        dataloader_1 = Dataloader('/home/aistudio/dataset', '/home/aistudio/dataset/val_list.txt', transform=preprocess, shuffle=True)
        val_load   = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        val_load.set_sample_generator(dataloader_1, batch_size=1, places=Place)
        model_dic, optic_dic = load_dygraph("./output/SETR-NotZero-Epoch-2-Loss-0.161517-MIOU-0.325002")
        model.load_dict(model_dic)
        model.eval()
        '''result = get_infer_data("/home/aistudio/dataset/infer")
        infer_load  = Load_infer('/home/aistudio/dataset', result, transform=preprocess, shuffle=False)
        loader_infer= fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        loader_infer.set_sample_generator(infer_load, batch_size=1, places=Place)
        process_image(model, loader_infer, result)'''
        validation(val_load, model, 15)



if __name__ == "__main__":
    main()