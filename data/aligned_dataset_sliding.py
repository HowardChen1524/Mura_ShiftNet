#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset

from PIL import Image
import cv2
import pandas as pd
import numpy as np

from util.utils_howard import tensor2img, enhance_img
ORISIZE = (512, 512)

class AlignedDatasetSliding(BaseDataset):
    def initialize(self, opt):
        self.opt = opt # param
        self.dir_A = opt.dataroot
        self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        print(f"Take all img: {len(self.A_paths)}")
        # crop stride 32, for training
        self.edge_index_list = [0, 105, 210, 14, 119, 224] 

        # crop stride 16, for find mura location
        self.corner_index_list = [0, 28, 812, 840]
        self.ud_index_list = [i for i in range(1, 28)] + [i for i in range(813, 840)]
        self.lr_index_list = [(29*i) for i in range(1, 27)] + [(28*i) for i in range(1, 27)]

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
        img_size = img.shape[:2] # h, w, c
        # if not 512,512 -> resize
        if self.opt.resolution == 'resized':
            img = cv2.resize(img, ORISIZE, interpolation=cv2.INTER_AREA)
        A = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        if self.opt.input_nc == 3:
            A = A.convert('RGB')
        else:
            A = A.convert('L')
            
        

        A_img = self.transform(A)
        # padding
        A_img  = A_img[:,6:-6,6:-6]
        # print(A_img.shape)
        up = torch.flip(A_img[:,:14,:], dims=[1])
        down = torch.flip(A_img[:,-14:,:], dims=[1])
        A_img = torch.concat((up, A_img, down), dim=1)

        # pil_img = tensor2img(A_img)
        # pil_img = enhance_img(pil_img)
        # pil_img.save(f"{A_path.split('/')[-1]}_ud.png")
        # print(A_img.shape)

        left = torch.flip(A_img[:,:,:14], dims=[2])
        right = torch.flip(A_img[:,:,-14:], dims=[2])
        A_img = torch.concat((left, A_img, right), dim=2)
        
        # pil_img = tensor2img(A_img)
        # pil_img = enhance_img(pil_img)
        # pil_img.save(f"{A_path.split('/')[-1]}_lr.png")
        print(A_img.shape)

        # pil_img = tensor2img(A_img)
        # pil_img = enhance_img(pil_img)
        # pil_img.save(f"{A_path.split('/')[-1]}.png")
        
        # sliding crop
        A_imgs = []
        c, h, w = A_img.size()
        # print(h,w)
        y_flag = False
        for y in range(0, h, self.opt.crop_stride): # stride default 32
            if y_flag: break
            crop_y = y
            if (y + self.opt.loadSize) >= h:
                y = h-self.opt.loadSize
                y_flag = True
            # print(f"y {y}")
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
            for i in range(0,225):
                if i not in self.edge_index_list:
                    crop_index_list.append(i)
            random.shuffle(crop_index_list)
            crop_index_list = crop_index_list[:self.opt.crop_image_num-len(self.edge_index_list)] + self.edge_index_list
            A_imgs = [A_imgs[crop_index] for crop_index in crop_index_list]
            
        A = torch.stack(A_imgs)
        # print(A.shape)
        
        mask = A.clone().zero_()
        
        # let B directly equals A
        B = A.clone()

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetSliding'
