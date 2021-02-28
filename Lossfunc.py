import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

def CostFunc(inputs, labels, ignore_index=255):
    n, c, h, w = inputs.shape
    n1,h1,w1,c1= labels.shape
    assert h==h1, "Shape Error"
    assert w==w1, "Shape Error"

    costfunc = fluid.layers.softmax_with_cross_entropy
    inputs   = fluid.layers.transpose(inputs, (0, 2, 3, 1))
    mask     = (labels != ignore_index)
    mask     = fluid.layers.cast(mask, 'float32')

    cost     = costfunc(inputs, labels)

    if fluid.layers.has_nan(cost):
        print("Error, there is nan in cost")
        exit()
    elif fluid.layers.has_inf(cost):
        print("Error, there is inf in cost")
        exit()
    
    cost = cost * mask

    avg_cost = fluid.layers.mean(cost) / (fluid.layers.mean(mask) + eps)

    return avg_cost

def main():
    label = np.random.randint(0, 13, [13, 255, 255, 1]).astype('int64')
    pred  = np.random.rand(13, 13, 255, 255).astype('float32')

    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred  = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss  = CostFunc(pred, label)
        print(loss)

if __name__ == "__main__":
    main()

