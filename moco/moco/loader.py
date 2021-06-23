# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import numpy as np
import cv2
import PIL
from torch.utils.data import Dataset
from PIL import Image
import os
from tqdm import tqdm
class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def extract_bbox(img) :
    ori_img = img.copy()
    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(img,100,255,0)
    _, contours, hierarchy = cv2.findContours(np.uint8(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 1 :
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
        x,y,w,h = cv2.boundingRect(biggest_contour)
        return ori_img[y:y+h,x:x+w,:]
    else:
        return ori_img
class RandomResize(object):
    def __init__(self):
        pass
    def __call__(self,x) :
        np.random.seed()
        max_size = 224
        bgr_img = np.uint8(np.zeros((max_size,max_size,3)))
        rand_size = np.random.choice(np.arange(120, 200, 20))
        img = np.array(x)
        ###
        # img = extract_bbox(img)
        # h,w,_ = img.shape
        # h_resize, w_resize = int(rand_size/max(h,w) * h), int(rand_size/max(h,w) * w)
        # img = cv2.resize(img,(w_resize,h_resize))
        ####
        img = cv2.resize(img, (rand_size,rand_size))
        h_resize, w_resize, _ = img.shape
        left_x = np.random.choice(np.arange(0, max_size - w_resize, 15))
        right_x = w_resize + left_x
        left_y = np.random.choice(np.arange(0, max_size - h_resize, 15))
        right_y = h_resize + left_y
        bgr_img[left_y:right_y,left_x:right_x,:] = img
        return PIL.Image.fromarray(np.uint8(bgr_img))

