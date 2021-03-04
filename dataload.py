import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid
import image_process

class Transform(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, input, label, flag = False):
        '''
         对于input图片，在完成resize工作之后还需要进行相应的归一化操作。
         否则在训练模型的时候会存在很大的误差。
         还有，label的数据类型必须转换成int64，否则paddle汇报很奇怪的错误。
        '''
        input = cv2.resize(input, (self.size, self.size), interpolation=cv2.INTER_LINEAR).astype('float32') / 255
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype("int64")
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] > 58:
                    flag = True
        return input, label, flag


class Dataloader():
    def __init__(self, image_path, list_file, transform = None, shuffle = False):
        self.image_path = image_path
        self.list_file  = list_file
        self.transform  = transform
        self.shuffle    = shuffle
        self.data_list  = self.read_list()

    def read_list(self):
        data_list = []
        with open(self.list_file, "r") as infile:
            for line in infile:
                data_path  = os.path.join(self.image_path, line.split()[0])
                label_path = os.path.join(self.image_path, line.split()[1])
                data_list.append((data_path, label_path))
            if self.shuffle:
                random.shuffle(data_list)
        return data_list

    def preprocess(self, input, label):
        h, w, c    = input.shape
        h1, w1     = label.shape
        assert h == h1, "Error shape!!"
        assert w == w1, "Error shape!!"
        if self.transform:
            input, label = self.transform(input, label)
            label        = label[:,:,np.newaxis]
        return input,label

    def __len__(self):
        return len(self.data_list)

    def __call__(self):
        '''
         使用yeild关键字的函数就是一个迭代器。迭代器的执行流程很像函数，也是顺序执行代码。
         迭代器的执行过程与普通函数的区别在于：迭代器没执行到一个yield语句就会终端，并返
         回一个值，而下次执行的时候则从yield语句开始执行。
        '''
        for data_path, label_path in self.data_list:
            input  = cv2.imread(data_path)
            input  = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            label  = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            input, label = self.preprocess(input, label)
            yield input, label
