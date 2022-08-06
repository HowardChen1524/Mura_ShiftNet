#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

import pandas as pd
import numpy as np
ORISIZE = 512

class AlignedDatasetSliding(BaseDataset):
    def initialize(self, opt):
        self.opt = opt # param
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A)) # return image path list (image_folder.py)
        if opt.continue_train:
            recover_list = []
            recover_df = pd.read_csv('/home/sallylab/Howard/shift-Net_sliding_crop/Mura_ShiftNet/training_imgs.csv')
            data_df = pd.read_csv('/home/levi/Howard/Mura/mura_data/RGB/0527_512/data_merged.csv')
            recover_fn = pd.merge(recover_df, data_df, on='PIC_ID', how='inner')['PIC_ID'].tolist()
            for fn in recover_fn:
                recover_list.append(f"{self.dir_A}{fn.replace('bmp','png')}")
            self.A_paths = sorted(recover_list)
            print(f"recover img num: {len(self.A_paths)}")
        # preprocessing
        if opt.isTrain:
            if self.opt.color_mode=='RGB':
                # pixel range -1~1
                transform_list = [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.RandomCrop(self.opt.fineSize)]
            else:
                transform_list = [transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5)),
                                transforms.RandomCrop(self.opt.fineSize)]
        else:
            if self.opt.color_mode=='RGB':
                transform_list = [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            else:
                transform_list = [transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))]
                                                    
        self.transform = transforms.Compose(transform_list)                                    

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert(self.opt.color_mode)

        # if not 512,512 -> resize
        if A.size != (ORISIZE, ORISIZE):
            A = A.resize((ORISIZE, ORISIZE), Image.BICUBIC)

        A_imgs = []
        if self.opt.isTrain:
            for i in range(self.opt.crop_image_num):
                A_imgs.append(self.transform(A))
            #     print(A_imgs[i].shape)
            #     print(A_imgs[i])
            # print(A_imgs[0] == A_imgs[1])
        else:
            A_img = self.transform(A)
            (c, w, h) = A_img.size()
            y_end_crop, x_end_crop = False, False
            for y in range(0, w, self.opt.crop_stride): # stride default 32
                # print(f"y {y}")

                y_end_crop = False
                
                for x in range(0, h, self.opt.crop_stride):
                    # print(f"x {x}")

                    x_end_crop = False

                    crop_y = y
                    if (y + self.opt.fineSize) > w:
                        crop_y =  w - self.opt.fineSize
                        y_end_crop = True

                    crop_x = x
                    if (x + self.opt.fineSize) > h:
                        crop_x = h - self.opt.fineSize
                        x_end_crop = True

                    crop_img = transforms.functional.crop(A_img, crop_y, crop_x, self.opt.fineSize, self.opt.fineSize)
                    A_imgs.append(crop_img)

                    if x_end_crop:
                        break

                if x_end_crop and y_end_crop:
                    break

        A = torch.stack(A_imgs)

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A.clone().zero_()
        
        # let B directly equals A
        B = A.clone()

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetSliding'
