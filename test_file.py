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

if __name__ == '__main__':
    Transformer_test()

    
