#-*-coding:utf-8-*-
import os
import math
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

from PIL import Image
import cv2
import pandas as pd
import numpy as np

from util.utils import tensor2img, enhance_img
W, H = 1920, 1080
EDGE_PIXEL = 6
PADDING_PIXEL = 14
class AlignedDatasetSliding(BaseDataset):
    def initialize(self, opt):
        self.opt = opt # param
        self.dir_A = opt.dataroot
        self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        print(f"Take all img: {len(self.A_paths)}")
        
        assert (self.opt.resolution == 'resized') or (self.opt.resolution == 'origin')
        if self.opt.resolution == 'resized':
            self.RESOLUTION = (512,512)
        else:
            self.RESOLUTION = (1920,1080)
            
        # crop stride 32, for training for 512
        # self.edge_index_list = [0, 105, 210, 14, 119, 224] 
        # crop stride 32, for training for 1920*1080
        # self.edge_index_list = [0, 1180, 1888, 58, 1238, 1946]
        self.num_w_crop = math.ceil((self.RESOLUTION[0]-self.opt.loadSize)/self.opt.crop_stride) + 1
        self.num_h_crop = math.ceil((self.RESOLUTION[1]-self.opt.loadSize)/self.opt.crop_stride) + 1
        self.edge_index_list = [0, self.num_w_crop*((self.num_h_crop//2)-1), self.num_w_crop*(self.num_h_crop-1), \
                                self.num_w_crop-1, (self.num_w_crop*(self.num_h_crop//2))-1, (self.num_w_crop*self.num_h_crop)-1] 
        
        if self.opt.input_nc==3:
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        else:
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5), (0.5))]
                                                                             
        self.transform = transforms.Compose(transform_list)                                    

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        img = cv2.imread(A_path)  
        
        assert (self.opt.resolution == 'resized') or (self.opt.resolution == 'origin')
        if self.opt.resolution == 'resized':
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        else:
            # 預設 blurring 錢處理
            ksize = (5, 5)  # 模糊核的大小
            sigmaX = 0      # X 方向的标准差
            sigmaY = 0      # Y 方向的标准差
            img = cv2.GaussianBlur(img, ksize, sigmaX, sigmaY)
            # pass
            
            
        A = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        # check blur image
        # b_dir = "./temp_img"
        # os.makedirs(b_dir, exist_ok=True)
        # pil_img = enhance_img(A)
        # pil_img.save(f"{b_dir}/{A_path.split('/')[-1]}.png")
        
        if self.opt.input_nc == 3:
            A = A.convert('RGB')
        else:
            A = A.convert('L')      

        A_img = self.transform(A)
        
        if (~self.opt.isTrain) and self.opt.isPadding == 1: 
            # pil_img = tensor2img(A_img)
            # pil_img = enhance_img(pil_img)
            # pil_img.save(f"{A_path.split('/')[-1]}_ori.png")

            A_img  = A_img[:,EDGE_PIXEL:-EDGE_PIXEL,EDGE_PIXEL:-EDGE_PIXEL]
            # pil_img = tensor2img(A_img)
            # pil_img = enhance_img(pil_img)
            # pil_img.save(f"{A_path.split('/')[-1]}_crop.png")

            up = torch.flip(A_img[:,:PADDING_PIXEL,:], dims=[1])
            down = torch.flip(A_img[:,-PADDING_PIXEL:,:], dims=[1])
            A_img = torch.concat((up, A_img, down), dim=1)
            # pil_img = tensor2img(A_img)
            # pil_img = enhance_img(pil_img)
            # pil_img.save(f"{A_path.split('/')[-1]}_ud.png")

            left = torch.flip(A_img[:,:,:PADDING_PIXEL], dims=[2])
            right = torch.flip(A_img[:,:,-PADDING_PIXEL:], dims=[2])
            A_img = torch.concat((left, A_img, right), dim=2)
            # pil_img = tensor2img(A_img)
            # pil_img = enhance_img(pil_img)
            # pil_img.save(f"{A_path.split('/')[-1]}.png")
        
        # sliding crop
        A_imgs = []
        _, h, w = A_img.size()
        # print(h, w)
        y_flag = False
        for y in range(0, h, self.opt.crop_stride): # stride default 32
            if y_flag: break
            crop_y = y
            if (y + self.opt.loadSize) >= h:
                crop_y = h-self.opt.loadSize
                y_flag = True
            x_flag = False
            for x in range(0, w, self.opt.crop_stride):
                if x_flag: break
                crop_x = x
                if (x + self.opt.loadSize) >= w:
                    crop_x = w-self.opt.loadSize
                    x_flag = True
                # print(f"x {x}")
                crop_img = transforms.functional.crop(A_img, crop_y, crop_x, self.opt.loadSize, self.opt.loadSize)
                A_imgs.append(crop_img)
        
        if self.opt.isTrain: 
            crop_index_list = []
            for i in range(0,len(A_imgs)):
                if i not in self.edge_index_list:
                    crop_index_list.append(i)
            random.shuffle(crop_index_list)
            crop_index_list = crop_index_list[:self.opt.crop_image_num-len(self.edge_index_list)] + self.edge_index_list
            A_imgs = [A_imgs[crop_index] for crop_index in crop_index_list]
            
        A = torch.stack(A_imgs)
        
        mask = A.clone().zero_()
        
        # let B directly equals A
        B = A.clone()

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetSliding'
