# -*- coding: utf-8 -*-
from __future__ import print_function
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np
from glob import glob
import os
import copy
from utils_gray import FDA_source_to_target_np, normalization
from PIL import Image, ImageFile, ImageFilter
from matplotlib import pyplot as plt
import random
import torch

np.random.seed(2)


class Fusionset(Data.Dataset):
    def __init__(self, args, root, transform=None, gray=False, partition='train', CR=True):
        self.files = glob(os.path.join(root, '*.*'))
        self.gray = gray
        self._tensor = transforms.ToTensor()
        self.transform = transform
        self.CR = CR
        self.args = args

        if args.miniset == True:
            self.files = random.sample(self.files, int(args.minirate * len(self.files)))
        self.num_examples = len(self.files)

        if self.CR == True:
            print('used CR')
        else:
            print('not used CR')

        if partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            np.random.shuffle(self.train_ind)
            self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
            np.random.shuffle(self.val_ind)
        print("number of " + partition + " examples in dataset" + ": " + str(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        # img = Image.open(self.files[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.gray:
            img = img.convert('L')
        # FORM here img.type is PIL.IMAGE
        img = np.array(img)  # 这里需要输入array，先不转tensor，但不必归一化至0-1之间

        if self.CR == True:
            sample_list = os.listdir(self.args.samplelist)
            sample_path = self.args.samplelist + random.sample(sample_list, 1)[0]
            sample = Image.open(sample_path)
            sample = np.array(sample)
            trans_img = FDA_source_to_target_np(img, sample, L=0.01)
            trans_img = np.round(trans_img)
            trans_img[trans_img > 255] = 255
            trans_img[trans_img < 0] = 0
            trans_img = trans_img.astype(np.uint8)
            # plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.imshow(img,cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(trans_img,cmap='gray')
            # plt.show()
            trans_img = self._tensor(trans_img)
            img = self._tensor(img)
        else:
            img = self._tensor(img)
            trans_img = img
        return img, trans_img


if __name__ == '__main__':
    samplelist = './dataset_scie/oe_resize_color/'
    sample_list = os.listdir(samplelist)
    sample_path = samplelist + random.sample(sample_list, 1)[0]
    print("")
