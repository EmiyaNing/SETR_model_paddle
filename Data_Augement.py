import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance

def normalize(image, mean, std):
    image = image.astype(np.float32, copy=False) / 255.0
    image -= mean
    image /= std
    return image

def resize(image, target_size=480, interp=cv2.INTER_LINEAR):
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        h = target_size[0]
        w = target_size[1]
    else:
        h = target_size
        w = target_size
    image = cv2.resize(image, (w, h), interpolation=interp)
    return image

def horizontal_flip(image):
    if len(image.shape) == 3:
        image = image[:, ::-3, :]
    elif len(image.shape) == 2:
        image = image[::-1, :]
    return image

def brightness(image, brightness_low, brightness_upper):
    brightness_delta = np.random.uniform(brightness_low, brightness_upper)
    image = Image.fromarray(image)
    image = ImageEnhance.Brightness(image).enhance(brightness_delta)
    image = np.array(image)
    return image

def contrast(image, contrast_lower, contrast_upper):
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    image = Image.fromarray(image)
    image = ImageEnhance.Contrast(image).enhance(contrast_delta)
    image = np.array(image)
    return image

def saturation(image, lower, upper):
    saturation_delta = np.random.uniform(lower, upper)
    image = Image.fromarray(image)
    image = ImageEnhance.Color(image).enhance(saturation_delta)
    image = np.array(image)
    return image

def hue(image, hue_lower, hue_upper):
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    image = Image.fromarray(image)
    image = np.array(image.convert('HSV'))
    image[:, :, 0] = image[:, :, 0] + hue_delta
    image = Image.fromarray(image, mode='HSV').convert('RGB')
    image = np.array(image)
    return image

def rotate(image, rotate_lower, rotate_upper):
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    image = Image.fromarray(image)
    image = image.rotate(int(rotate_delta))
    image = np.array(image)
    return image


def center_crop(image, crop_size):
    center_h = image.shape[0] // 2
    center_w = image.shape[1] // 2
    image = image[center_h-crop_size//2:center_h+crop_size//2, center_w-crop_size//2:center_w+crop_size//2]
    return image

def random_crop(image, crop_size):
    if len(image.shape) == 3:
        hight, width, depth = image.shape
    elif len(image.shape) == 2:
        hight, width = image.shape
    limit_h = hight - crop_size
    limit_w = width - crop_size
    start_h = random.randint(0, limit_h-1)
    start_w = random.randint(0, limit_w-1)
    image   = image[start_h:start_h+crop_size, start_w:start_w+crop_size]
    return image, start_h, start_w


class Data_Preprocess(object):
    def __init__(self, size, mean_val = 0 ,std_val = 1):
        self.crop_size = size
        self.mean_val  = mean_val
        self.std_val   = std_val

    def __call__(self, image, label, flag=False):
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] > 58:
                    flag = True
        # First choose resize or center_crop or random_crop
        id_num = random.randint(0, 2)
        if id_num == 0:
            image = resize(image, self.crop_size)
            label = resize(label, self.crop_size, interp=cv2.INTER_NEAREST)
        elif id_num == 1:
            image = center_crop(image, self.crop_size)
            label = center_crop(label, self.crop_size)
        elif id_num == 2:
            image,sh,sw = random_crop(image, self.crop_size)
            label = label[sh:sh+self.crop_size, sw:sw+self.crop_size]
        # Second choose a augment way, horizontal_flip, brightness, contrast, saturation, hue
        way_num = random.randint(0, 4)
        if way_num == 0:
            image = horizontal_flip(image)
            label = horizontal_flip(image)
        elif way_num == 1:
            image = brightness(image, 0, 10)
        elif way_num == 2:
            image = contrast(image, 0, 10)
        elif way_num == 3:
            image = saturation(image, 0 ,10)
        elif way_num == 4:
            image = hue(image, 0, 10)
        # Thirdly normize the result
        image = normalize(image, self.mean_val, self.std_val).astype('float32')
        label = label.astype('int64')
        return image, label,flag