from __future__ import print_function
import os
import sys
import cv2
import random
import math
import time
import scipy.io
import numpy as np
import torch
import torch.utils.data as data
from Config import *
from Utils import rotate_and_crop


class ColorChecker(data.Dataset):
    def __init__(self, train=True, folds_num=1):
        list_path = "../../../dataset/color_checker/color_cheker_data_meta.txt"
        with open(list_path, 'r') as f:
            self.all_data_list = f.readlines()
        self.data_list = []
        folds = scipy.io.loadmat("../../../dataset/color_checker/folds.mat")
        if train:
            img_idx = folds['tr_split'][0][folds_num][0]
            for i in img_idx:
                self.data_list.append(self.all_data_list[i-1])
        else:
            img_idx = folds['te_split'][0][folds_num][0]
            for i in img_idx:
                self.data_list.append(self.all_data_list[i-1])
        self.train = train

    def __getitem__(self, index):
        model = self.data_list[index]
        # filename
        fn = model.strip().split(' ')[1]
        img = np.load("../../../dataset/color_checker/npy/ndata/"+fn+'.npy')
        illums = np.load("../../../dataset/color_checker/npy/nlabel/"+fn+'.npy')
        img = np.array(img,dtype='float32')
        illums = np.array(illums,dtype='float32')
        if self.train:
            img, illums = self.augment_train(img,illums)
            img = np.clip(img, 0.0, 65535.0)
            img = img * (1.0 / 65535)
            img = img[:,:,::-1] # BGR to RGB
            img = np.power(img,(1.0/2.2))
            img = img.transpose(2,0,1) # hwc to chw
            img = torch.from_numpy(img.copy())
            illums = torch.from_numpy(illums.copy())
        else:
            img = self.crop_test(img,illums)
            img = np.clip(img, 0.0, 65535.0)
            img = img * (1.0 / 65535)
            img = img[:,:,::-1] # BGR to RGB
            img = np.power(img,(1.0/2.2))
            img = img.transpose(2,0,1) # hwc to chw
            img = torch.from_numpy(img.copy())
            illums = torch.from_numpy(illums.copy())
            img = img.type(torch.FloatTensor)

        return img,illums,fn

    def augment_train(self,ldr, illum):
        angle = (random.random() - 0.5) * AUGMENTATION_ANGLE
        scale = math.exp(random.random() * math.log(AUGMENTATION_SCALE[1] / AUGMENTATION_SCALE[0])) * AUGMENTATION_SCALE[0]
        s = int(round(min(ldr.shape[:2]) * scale))
        s = min(max(s, 10), min(ldr.shape[:2]))
        start_x = random.randrange(0, ldr.shape[0] - s + 1)
        start_y = random.randrange(0, ldr.shape[1] - s + 1)
        flip_lr = random.randint(0, 1) # Left-right flip?
        color_aug = np.zeros(shape=(3, 3))
        for i in range(3):
            color_aug[i, i] = 1 + random.random() * AUGMENTATION_COLOR - 0.5 * AUGMENTATION_COLOR

        def crop(img, illumination):
            if img is None:
                return None
            img = img[start_x:start_x + s, start_y:start_y + s]
            img = rotate_and_crop(img, angle)
            img = cv2.resize(img, (FCN_INPUT_SIZE, FCN_INPUT_SIZE))
            if flip_lr:
                img = img[:, ::-1]
            img = img.astype(np.float32)
            new_illum = np.zeros_like(illumination)
            # RGB -> BGR
            illumination = illumination[::-1]
            for i in range(3):
                for j in range(3):
                    new_illum[i] += illumination[j] * color_aug[i, j]

            img *= np.array([[[color_aug[0][0], color_aug[1][1], color_aug[2][2]]]],dtype=np.float32)
            new_image = img
            new_image = np.clip(new_image, 0, 65535)
            new_illum = np.clip(new_illum, 0.01, 100)
            return new_image, new_illum[::-1]
        return crop(ldr, illum)

    def crop_test(self,img,illums,scale=0.5):
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
        return img

    def __len__(self):
        return(len(self.data_list))
def size_finetuning(h_w):
    # check the image_size for the ability that the deconvolution function can output the same image size
    for i in range(2):

        while True:
            k = h_w[i]
            #conv and maxpool
            k = math.floor((k - 3)/2 + 1)
            k = math.ceil((k - 3)/2 + 1)
            k = math.ceil((k - 3) / 2 + 1)
            k = math.ceil((k - 3) / 2 + 1)
            #deconv
            k = (k - 1) * 2 + 3
            k = (k - 1) * 2 + 3
            k = (k - 1) * 2 + 3
            k = (k - 1) * 2 + 4

            if k != h_w[i]:
                h_w[i] -= 1
            else:
                break
    return h_w

def img_size_finetuning(img):
    h_w = img.shape[2:]
    h_w = torch.tensor(h_w)
    h_w = size_finetuning(h_w)
    return img[:, :, :h_w[0], :h_w[1]]




if __name__=='__main__':
    img = torch.randn(2,3,730,730)
    img = img_size_finetuning(img)

