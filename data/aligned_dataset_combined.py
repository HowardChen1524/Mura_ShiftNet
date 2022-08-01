#-*-coding:utf-8-*-
import os.path
from random import shuffle
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from collections import defaultdict
import pandas as pd

class AlignedDatasetCombined(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.s_dir_A = opt.testing_smura_dataroot
        self.n_dir_A = opt.testing_normal_dataroot
        
        self.s_A_paths = sorted(make_dataset(self.s_dir_A))
        self.n_A_paths = sorted(make_dataset(self.n_dir_A))
        shuffle(self.n_A_paths)
        self.n_A_paths = self.n_A_paths[:len(self.s_A_paths)]
        
        self.w = 512
        self.h = 512

        # preprocessing
        if self.opt.color_mode == 'RGB':
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        else:
            transform_list = [transforms.ToTensor(), 
                            transforms.Normalize((0.5),(0.5))]

        self.transform = transforms.Compose(transform_list)
        
        # type-c
        self.smura_pos_df = pd.read_csv('./MURA_XY.csv')
        self.bounding_box = opt.loadSize
        
    def __getitem__(self, index):
        data_dict = defaultdict(dict)
        type_dict = {'smura': self.s_A_paths, 'normal': self.n_A_paths}
        
        crop_x, crop_y = 0, 0 # for normal
        for type_name, A_paths in type_dict.items(): # key, value
            A_path = A_paths[index]
            A = Image.open(A_path).convert(self.opt.color_mode) # W * H * C    
            # if not 512,512 -> resize
            if A.size != (self.w, self.h):
                A = A.resize((self.w, self.h), Image.BICUBIC)
        
            A = self.transform(A) # N x C x H x W
            # crop_x -> w -> col
            # crop_y -> h -> row
            if type_name == 'smura':
                fn = A_path[len(self.s_dir_A):].replace('png','bmp')
                fn_series = self.smura_pos_df[self.smura_pos_df['PIC_ID'] == fn]

                if fn_series['PRODUCT_CODE'].values[0] == 'T850QVN03': # 4k
                    crop_x = int(fn_series['X'] / 2 / 3.75) - (self.bounding_box // 2)
                    crop_y = int(fn_series['Y'] / 2 / 2) - (self.bounding_box // 2)
                elif fn_series['PRODUCT_CODE'].values[0] == 'T850MVR05': # 8k
                    crop_x = int(fn_series['X'] / 4 / 3.75) - (self.bounding_box // 2)
                    crop_y = int(fn_series['Y'] / 4 / 2) - (self.bounding_box // 2)
                    
                if crop_x < 0:
                    crop_x = 0
                    print(f'{fn} -> x < 0')
                if crop_y < 0:
                    crop_y = 0
                    print(f'{fn} -> y < 0')
                
                if (crop_x + self.bounding_box) > self.w:
                    crop_x = 511 - self.opt.fineSize
                    print(f'{fn} -> x > 511')
                if (crop_y + self.bounding_box) > self.h:
                    crop_y =  511 - self.opt.fineSize
                    print(f'{fn} -> y > 511')
                
                A = transforms.functional.crop(A, crop_y, crop_x, self.bounding_box, self.bounding_box)
                # print(f'x: {crop_x}')
                # print(f'y: {crop_y}')
            else:
                # print(f'x: {crop_x}')
                # print(f'y: {crop_y}')
                A = transforms.functional.crop(A, crop_y, crop_x, self.bounding_box, self.bounding_box)
            # Just zero the mask is fine if not offline_loading_mask.
            mask = A.clone().zero_()

            # let B directly equals A
            B = A.clone()

            data_dict[type_name] = {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}
        return data_dict

    def __len__(self):
        # return {'normal': len(self.n_A_paths), 'smura': len(self.s_A_paths)}
        return len(self.n_A_paths)

    def name(self):
        return 'AlignedDatasetCombined'
