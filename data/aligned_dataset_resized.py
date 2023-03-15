#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import cv2

ORISIZE = (512, 512)

class AlignedDatasetResized(BaseDataset):
    def initialize(self, opt):
        self.opt = opt # param
        self.dir_A = opt.dataroot
        self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        print(f"Take all img: {len(self.A_paths)}")

        # preprocessing
        if self.opt.color_mode == 'RGB':
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        else:
            transform_list = [transforms.ToTensor(), 
                            transforms.Normalize((0.5),(0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # A_path = self.A_paths[index]
        # A = Image.open(A_path).convert(self.opt.color_mode)

        # # if not 512,512 -> resize
        # if A.size != (ORISIZE, ORISIZE):
        #     A = A.resize((ORISIZE, ORISIZE), Image.BICUBIC)

        A_path = self.A_paths[index]
        img = cv2.imread(A_path)  
        # if not 512,512 -> resize
        
        assert (self.opt.resolution == 'resized') or (self.opt.resolution == 'origin')
        if self.opt.resolution == 'resized':
            img = cv2.resize(img, ORISIZE, interpolation=cv2.INTER_AREA)
        A = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  
        A = A.convert(self.opt.color_mode)

        A = self.transform(A)

        mask = A.clone().zero_()

        # let B directly equals A
        B = A.clone()

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetResized'
