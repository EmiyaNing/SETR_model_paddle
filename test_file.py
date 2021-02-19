from models import *
import paddle
import paddle.fluid
import numpy as np

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


if __name__ == '__main__':
    Encoder_test()

    