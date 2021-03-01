from models import *
import paddle
import paddle.fluid as fluid
import numpy as np
import Data_Augement as augment
from Data_Augement import Data_Preprocess
from dataload import Dataloader,Transform

def Embedding_test():
    with fluid.dygraph.guard():
        image = np.random.rand(13, 3, 256, 256).astype('float32')
        image = fluid.dygraph.to_variable(image)
        embedding_func = Embedding(1024, 256, 3, 16, 0.8)
        result = embedding_func(image)
        print(result.shape)
    

def Embedding_Attention_test():
    with fluid.dygraph.guard():
        image = np.random.rand(13, 3, 256, 256).astype('float32')
        image = fluid.dygraph.to_variable(image)
        embedding_func = Embedding(1024, 256, 3, 16, 0.8) 
        attention      = Attention(1024, 8, 0.8)
        result = embedding_func(image)
        print(result.shape)
        out, weight = attention(result)
        print(out.shape)
        print(weight.shape)

def Embedding_Attention_MLP_test():
    with fluid.dygraph.guard():
        image = np.random.rand(13, 3, 256, 256).astype('float32')
        image = fluid.dygraph.to_variable(image)
        embedding_func = Embedding(1024, 256, 3, 16, 0.8) 
        attention      = Attention(1024, 8, 0.8)
        fully_con      = MLP(1024, 0.8)
        result = embedding_func(image)
        out, weight = attention(result)
        res    = fully_con(out)
        print(res.shape)
        

def Block_test():
    with fluid.dygraph.guard():
        image = np.random.rand(13, 3, 256, 256).astype('float32')
        image = fluid.dygraph.to_variable(image)
        embedding_func = Embedding(1024, 256, 3, 16, 0.8) 
        block = Block(1024, 8, 0.8)
        result= embedding_func(image)
        res, weight = block(result)
        print(res.shape)
        print(weight.shape)

def Encoder_test():
    with fluid.dygraph.guard():
        image = np.random.rand(13, 3, 256, 256).astype('float32')
        image = fluid.dygraph.to_variable(image)
        embedding_func = Embedding(1024, 256, 3, 16, 0.8) 
        encoder = Encoder(1024, 6, 8, 0.8, True)
        embedding_position = embedding_func(image)
        res, weight = encoder(embedding_position)
        print(res.shape)
        for element in weight:
            print(element.shape)


def Decoder_Naive_test():
    with fluid.dygraph.guard():
        data = np.random.rand(13, 257, 1024).astype('float32')
        data = fluid.dygraph.to_variable(data)
        decoder = Decoder_Naive(13, 1024, 256, 256, 0.8)
        res  = decoder(data)
        print(res.shape)

def Decoder_PUP_test():
    with fluid.dygraph.guard():
        data = np.random.rand(13, 257, 1024).astype('float32')
        data = fluid.dygraph.to_variable(data)
        decoder = Decoder_PUP(13, 1024, 256, 0.8)
        res  = decoder(data)
        print(res.shape)


def Decoder_MLA_test():
    with fluid.dygraph.guard():
        input = []
        for i in range(4):
            data = np.random.rand(13, 257, 1024).astype('float32')
            data = fluid.dygraph.to_variable(data)
            input.append(data)
        decoder = Decoder_MLA(13, 1024, 256, 0.8)
        res     = decoder(input)
        print(res.shape)

def Transformer_test():
    with fluid.dygraph.guard():
        image = np.random.rand(13, 3, 256, 256).astype('float32')
        image = fluid.dygraph.to_variable(image)
        transformer = Transformer(256, 13, 1024, 8, 8, 0.8, 'Naive')
        res, weight = transformer(image)
        print(res.shape)
        for element in weight:
            print(element.shape)

def Augment_test():
    image = cv2.imread("test.jpg")
    norm  = augment.normalize(image, 10, 1)
    resize= augment.resize(image, 480)
    horiz = augment.horizontal_flip(image)
    bright= augment.brightness(image, 50, 200)
    contrast = augment.contrast(image, 10, 200)
    satura= augment.saturation(image, 10, 200)
    hue   = augment.hue(image, 10, 200)
    rotate= augment.rotate(image, 10, 200)
    center= augment.center_crop(image, 480)
    random,h,w= augment.random_crop(image, 480)
    cv2.imwrite('norm.jpg', norm)
    cv2.imwrite('resize.jpg', resize)
    cv2.imwrite('horize.jpg', horiz)
    cv2.imwrite('bright.jpg', bright)
    cv2.imwrite('contrast.jpg', contrast)
    cv2.imwrite('saturate.jpg', satura)
    cv2.imwrite('hue.jpg', hue)
    cv2.imwrite('rotate.jpg', rotate)
    cv2.imwrite('center.jpg', center)
    cv2.imwrite('random.jpg', random)

def DataAugment_test():
    preprocess = Data_Preprocess(480, 0 ,1)
    image      = cv2.imread('test.jpg')
    label      = np.random.rand(480, 480, 1)
    res1, res2 = preprocess(image, label)
    cv2.imwrite('rand_class.jpg', res1)

def Dataloader_test():
    Place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(Place):
        #preprocess = Data_Preprocess(480, 0, 1)
        preprocess = Transform(480)
        dataloader = Dataloader("/home/aistudio/data/data68698", "/home/aistudio/data/data68698/train_list.txt", transform=preprocess, shuffle=True)
        Loader     = fluid.io.DataLoader.from_generator(capacity=1, use_multiprocess=False)
        Loader.set_sample_generator(dataloader, batch_size = 8, places = Place)
        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(Loader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape: {label.shape}')

if __name__ == '__main__':
    Dataloader_test()

    
