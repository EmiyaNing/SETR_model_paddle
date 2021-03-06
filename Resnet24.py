import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Conv2D
from paddle.fluid.dygraph import BatchNorm
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Linear
from paddle.fluid.dygraph import Dropout

class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 dilation=1,
                 padding=None,
                 name=None):
        super(ConvBNLayer, self).__init__(name)

        if padding is None:
            padding = (filter_size-1)//2
        else:
            padding=padding

        self.conv = Conv2D(num_channels=num_channels,
                            num_filters=num_filters,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            act=None,
                            dilation=dilation,
                            bias_attr=False)
        self.bn = BatchNorm(num_filters, act=act)

    def forward(self, inputs):
        y = self.conv(inputs)
        y = self.bn(y)
        return y


class BasicBlock(fluid.dygraph.Layer):
    expansion = 1  # expand ratio for last conv output channel in each block
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride=1,
                 shortcut=True,
                 name=None):
        super(BasicBlock, self).__init__(name)
        
        self.conv0 = ConvBNLayer(num_channels=num_channels,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 stride=stride,
                                 act='brelu',
                                 name=name)
        self.conv1 = ConvBNLayer(num_channels=num_filters,
                                 num_filters=num_filters,
                                 filter_size=3,
                                 act='brelu',
                                 name=name)
        if not shortcut:
            self.short = ConvBNLayer(num_channels=num_channels,
                                     num_filters=num_filters,
                                     filter_size=1,
                                     stride=stride,
                                     act=None,
                                     name=name)
        self.shortcut = shortcut

    def forward(self, inputs):
        conv0 = self.conv0(inputs)
        conv1 = self.conv1(conv0)
        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=conv1, act='relu')
        return y




class ResNet24(fluid.dygraph.Layer):
    '''
        This model is used to decrease the size of input image
        1. Decrease the image size 16 time
        2. Add s
    '''
    def __init__(self, hidden_unit_num, image_size, in_channel, patch_num, dropout):
        super(ResNet24, self).__init__()
        self.patch_size = image_size // patch_num
        n_patch_size    = (image_size // patch_num) * (image_size // patch_num)
        self.position_embedding = fluid.layers.create_parameter((1, n_patch_size+1, hidden_unit_num), dtype='float32', is_bias=True)
        self.cls_token          = fluid.layers.create_parameter((1, 1, hidden_unit_num), is_bias=False, dtype='float32')
        self.dropout            = Dropout(dropout)


        self.layer1_1 = BasicBlock(num_channels=in_channel,
                                   num_filters=24,
                                   stride=1,
                                   shortcut=False)
        self.layer1_2 = BasicBlock(num_channels=24,
                                   num_filters=48,
                                   stride=2,
                                   shortcut=False)
        self.layer2_1 = BasicBlock(num_channels=48,
                                   num_filters=48,
                                   stride=1,
                                   shortcut=False)
        self.layer2_2 = BasicBlock(num_channels=48,
                                   num_filters=48,
                                   stride=2,
                                   shortcut=False)                           
        self.layer3_1 = BasicBlock(num_channels=48,
                                   num_filters=128,
                                   stride=1,
                                   shortcut=False)
        self.layer3_2 = BasicBlock(num_channels=128,
                                   num_filters=256,
                                   stride=2,
                                   shortcut=False)
        self.layer4_1 = BasicBlock(num_channels=256,
                                   num_filters=256,
                                   stride=1,
                                   shortcut=False)
        self.layer4_2 = BasicBlock(num_channels=256,
                                   num_filters=hidden_unit_num,
                                   stride=2,
                                   shortcut=False)


    def forward(self,input):
        num_input = input.shape[0]
        cls_token = fluid.layers.expand(self.cls_token, expand_times=[num_input, 1, 1])
        x = self.layer1_1(input)
        x = self.layer1_2(x)
        x = self.layer2_1(x)
        x = self.layer2_2(x)
        x = self.layer3_1(x)
        x = self.layer3_2(x)
        x = self.layer4_1(x)
        x = self.layer4_2(x)
        x = fluid.layers.reshape(x, [x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        x = fluid.layers.transpose(x, (0, 2, 1))
        x = fluid.layers.concat(input=[cls_token, x], axis=1)
        embeddings = x + self.position_embedding
        embeddings = self.dropout(embeddings)
        return embeddings

if __name__ == "__main__":
    with fluid.dygraph.guard():
        x_data = np.random.rand(2, 3, 480, 480).astype(np.float32)
        x = to_variable(x_data)
        model = ResNet24(hidden_unit_num=1024,
                         image_size=480,
                         in_channel=3,
                         patch_num=16,
                         dropout=0.8)
        pred  = model(x)
        print(pred.shape)