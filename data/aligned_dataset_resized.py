#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

ORISIZE = 512

class AlignedDatasetResized(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A))

        # preprocessing
        if self.opt.color_mode == 'RGB':
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        else:
            transform_list = [transforms.ToTensor(), 
                            transforms.Normalize((0.5),(0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert(self.opt.color_mode)

        # if not 512,512 -> resize
        if A.size != (ORISIZE, ORISIZE):
            A = A.resize((ORISIZE, ORISIZE), Image.BICUBIC)

        A = self.transform(A)

        mask = A.clone().zero_()

        # let B directly equals A
        B = A.clone()

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetResized'
