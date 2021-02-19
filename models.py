'''
    This file define a transformer class.
    1. This model is used to finish image segmentation task
    2. The basic framework is baidu's paddlepaddle 1.8
    3. Transformer is consist by Decoder class and Encoder class
    4. Encoder is consist by transformer model(self multi-head attention model, Fully Connect network)
    5. Decoder can be set as traditional segmentation upsampling(1*1 Conv + bilinear upsampling, processing upsampling, etc).
    6. This transformer model will divide a image with 480*480*3 to a grid 30 * 30 * 3 image
    7. Each 30 * 30 * 3 sub-image will reshape as 900 * 3 vector. 
    8. All 900 * 3 vector will used to concat a sequence...
'''
import cv2
import copy
import paddle
import math
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.dygraph import Layer
from paddle.fluid.dygraph import LayerNorm
from paddle.fluid.dygraph import Dropout
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import LayerList

class Attention(Layer):
    '''
        In this class we will implement the self-attention function...
    '''
    def __init__(self, 
                 hidden_unit_num,           # the fully connect layer's unit number
                 head_num,                  # self-attention's head number
                 dropout,                   # dropout rate...   
                 visualable=True):          # whether show the attention value                   
        '''
            
        '''
        super(Attention, self).__init__()
        self.vis  = visualable
        self.num_attention_head  = head_num
        self.attention_head_size = int(hidden_unit_num / head_num)
        self.all_head_size       = self.num_attention_head * self.attention_head_size

        self.query = Linear(hidden_unit_num, self.all_head_size)
        self.key   = Linear(hidden_unit_num, self.all_head_size)
        self.value = Linear(hidden_unit_num, self.all_head_size)
        self.output= Linear(hidden_unit_num, hidden_unit_num)

        self.atte_dropout = Dropout(dropout)
        self.proj_dropout = Dropout(dropout)
        self.softmax      = fluid.layers.softmax
        
    def transpose_for_score(self,x):
        new_shape = x.shape[:-1] + [self.num_attention_head, self.attention_head_size]
        x         = fluid.layers.reshape(x, new_shape)
        res       = fluid.layers.transpose(x, [0, 2, 1, 3])
        return res

    def forward(self, input):
        mix_query = self.query(input)
        mix_key   = self.key(input)
        mix_value = self.value(input)


        query_layer = self.transpose_for_score(mix_query)
        key_layer   = self.transpose_for_score(mix_key)
        value_layer = self.transpose_for_score(mix_value)


        attention_score = fluid.layers.matmul(query_layer, fluid.layers.transpose(key_layer,[0,1,3,2]))
        attention_score = attention_score / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_score)

        if self.vis == True:
            weight = attention_probs
        else:
            weight = None
        
        attention_probs = self.atte_dropout(attention_probs)
        
        contex_layer = fluid.layers.matmul(attention_probs, value_layer)
        contex_layer = fluid.layers.transpose(contex_layer, [0, 2, 1, 3])
        contex_layer = fluid.layers.reshape(contex_layer, [contex_layer.shape[0], contex_layer.shape[1], contex_layer.shape[2] * contex_layer.shape[3]])
        attention_out = self.output(contex_layer)
        attention_out = self.proj_dropout(attention_out)
        return attention_out, weight




class MLP(Layer):
    '''
        In this class we will implement the transformer's fully connected layer...
    '''
    def __init__(self,
                 hidden_unit_num,           # the fully connect layer's unit number
                 dropout):                  # dropout rate.....
        super(MLP, self).__init__()
        self.fc1 = Linear(hidden_unit_num, 3072, bias_attr=True)
        self.fc2 = Linear(3072, hidden_unit_num, bias_attr=True)
        self.act = fluid.layers.gelu
        self.dropout = Dropout(dropout)
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
        


class Embedding(Layer):
    '''
        In this class we will implemnt the image divide operator..
    '''
    def __init__(self,
                 hidden_unit_num,           # hidden layer's unit number 
                 image_size,                # the input image size
                 in_channel,              # the input image channel
                 patch_num,                 # how much sub-image in a input image..
                 dropout):                  # dropout rate......
        super(Embedding, self).__init__()
        self.patch_size = image_size // patch_num                          
        n_patch_size    = (image_size // patch_num) * (image_size // patch_num)
        self.patch_embedding    = Conv2D(num_channels=in_channel,
                                      num_filters=hidden_unit_num,
                                      filter_size=self.patch_size,
                                      stride=self.patch_size)
        self.position_embedding = fluid.layers.create_parameter((1, n_patch_size+1, hidden_unit_num),
                                                                dtype='float32',
                                                                is_bias=True)
        self.cls_token          = fluid.layers.create_parameter((1, 1, hidden_unit_num), is_bias=False, dtype='float32')
        self.dropout            = Dropout(dropout)


    def forward(self, input):
        '''
            傻逼paddle的flatten函数只能将高维tensor拉平成2维tensor.
            我觉得写这个函数的开发人员脑子绝对有问题，你娘的谁特么一定样将一个tensor拉平成2维呢。
            我是李彦宏我特么绝对把这个程序员给开除掉。
        '''
        num_input = input.shape[0]
        cls_token = fluid.layers.expand(self.cls_token, expand_times=[num_input, 1, 1])
        x         = self.patch_embedding(input)
        x         = fluid.layers.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        x         = fluid.layers.transpose(x, (0, 2, 1))
        x         = fluid.layers.concat(input=[cls_token, x], axis=1)
        embeddings= x + self.position_embedding
        embeddings= self.dropout(embeddings)
        return embeddings



class Block(Layer):
    '''
        In this class we will construct a transformer block. used to construct network...
    '''
    def __init__(self, hidden_unit_num, head_num, dropout, visualable=True):
        super(Block, self).__init__()
        self.attention_norm = LayerNorm(hidden_unit_num, epsilon=1e-6)
        self.fully_con_norm = LayerNorm(hidden_unit_num, epsilon=1e-6)
        self.fully_connect  = MLP(hidden_unit_num, dropout)
        self.attention_layer= Attention(hidden_unit_num, 8, dropout, visualable)

    def forward(self, input):
        h = input
        # Now the batch norm have some bug...
        x = self.attention_norm(input)
        x, weight = self.attention_layer(x)
        x = h + x

        h = x
        x = self.fully_con_norm(x)
        x = self.fully_connect(x)
        x = x + h
        return x, weight



class Encoder(Layer):
    '''
        Top level Encoder class...
        In this class, the input image will be divide as a grid of patches. 
        Each patch is a vector whose length is h*w/256.
        This class is consist by transformer modul.
    '''
    def __init__(self, 
                 hidden_unit_num,
                 layer_num,
                 head_num,
                 dropout,
                 visualable):
        super(Encoder, self).__init__()
        self.vis  = visualable
        self.model= LayerList()
        self.encoder_norm = LayerNorm(hidden_unit_num, epsilon=1e-6)
        for i in range(layer_num):
            layer = Block(hidden_unit_num, head_num, dropout, visualable)
            self.model.append(layer)

    
    def forward(self, input):
        attention_weight = []
        for layer in self.model:
            input , weight = layer(input)
            if self.vis:
                attention_weight.append(weight)
        encoded = self.encoder_norm(input)
        return encoded, attention_weight



class Decoder(Layer):
    '''
        Top level Decoder class...
        In this class, the input line list will be reshape as a image list, and each image's size is h/16 * w/16.
        Decode way : (1) Naive upsampling:1 x 1 conv + sync batch norm + 1 x 1 conv + bilinearly upsample
                     (2) Progressive Upsampling: progressive upsampling, Just like U2Net up sample way.
                     (3) Multi-Level feature Aggregation: 
    '''
    def __init__(self,):
        super(Decoder, self).__init__()
    
    def forward(self,):
        pass


class Transformer(Layer):
    '''
        Top level class.....
    '''
    def __init__(self,  
                 image_size=480,        # the input image size
                 num_classes=59,        # the class of the segmentation image... 
                 hidden_unit_num=2048,   # the fully connect layer's unit number
                 layer_num=6,           # the fully connect layer's number
                 head_num=8,            # the number of self-attention head
                 dropout=0.1,           # the dropout probility
                 visualable=True):
        super(Transformer, self).__init__()
        self.embedding  = Embedding(hidden_unit_num, image_size, 3, 16, dropout)
        self.encoder = Encoder(image_size, hidden_unit_num,layer_num, head_num, dropout, visualable)
        self.decoder = Decoder(num_classes, hiden_unit_num, dropout)

    def forward(self, input):
        '''
            The input image will be process by a encoder class. 
            Encoder class is consist by transformer modul.
            And then the result of encoder will be passed to Decoder class to form the output image.
            Decoer class is consist by Upsampling convolution...
        '''
        
        x = self.embedding(input)
        x, weight = self.encoder(x)
        x = self.decoder(x)
        return x

    