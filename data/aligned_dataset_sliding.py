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

ORISIZE = (512, 512)

class AlignedDatasetSliding(BaseDataset):
    def initialize(self, opt):
        self.opt = opt # param
        self.dir_A = opt.dataroot
        self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        print(f"Take all img: {len(self.A_paths)}")
        self.edge_index_list = [0, 105, 210, 14, 119, 224]
        # Deprecate random choose
        # # train
        # if (opt.isTrain) and (not opt.continue_train):
        #     self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        #     random.shuffle(self.A_paths) # shuffle path list

        #     self.A_paths = self.A_paths[:opt.random_choose_num]

        #     # save filename
        #     recover_list = []
        #     for i, path in enumerate(self.A_paths):
        #         # print(i)
        #         recover_list.append(path[len(self.dir_A):].replace('.png','.bmp'))
        #     recover_df = pd.DataFrame(recover_list, columns=['PIC_ID'])
        #     recover_df.to_csv('./training_imgs.csv', index=False, columns=['PIC_ID'])
        #     print(f"Record {len(self.A_paths)} filename successful!")
        # # continue train
        # elif (opt.isTrain) and (opt.continue_train):
        #     self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        #     random.shuffle(self.A_paths) # shffle path list
        #     self.A_paths = self.A_paths[:opt.random_choose_num]

        #     # save filename
        #     expr_dir = os.path.join(opt.checkpoints_dir, opt.model_version)
        #     recover_list = []
        #     for i, path in enumerate(self.A_paths):
        #         # print(i)
        #         recover_list.append(path[len(self.dir_A):].replace('.png','.bmp'))
        #     recover_df = pd.DataFrame(recover_list, columns=['PIC_ID'])
        #     recover_df.to_csv(os.path.join(expr_dir, 'training_imgs.csv'), index=False, columns=['PIC_ID'])
        #     print(f"Record {len(self.A_paths)} filename successful!")

        #     recover_list = []
        #     recover_df = pd.read_csv('./training_imgs.csv')
        #     data_df = pd.read_csv('/home/levi/mura_data/d17/data_merged.csv')
        #     recover_fn = pd.merge(recover_df, data_df, on='PIC_ID', how='inner')['PIC_ID'].tolist()
        #     for fn in recover_fn:
        #         recover_list.append(f"{self.dir_A}{fn.replace('bmp','png')}")
        #     self.A_paths = recover_list
        #     print(f"Recover img num: {len(self.A_paths)}")
        # # test
        # else:
        #     self.A_paths = make_dataset(self.dir_A) # return image path list (image_folder.py)
        #     print(f"Take all img: {len(self.A_paths)}")
        
        # preprocessing
        # if opt.isTrain:
        #     if self.opt.color_mode=='RGB':
        #         transform_list = [transforms.ToTensor(),
        #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # pixel range -1~1
        #                         transforms.RandomCrop(self.opt.fineSize)]
        #     else:
        #         transform_list = [transforms.ToTensor(),
        #                         transforms.Normalize((0.5), (0.5)),
        #                         transforms.RandomCrop(self.opt.fineSize)]
        # else:
        #     if self.opt.color_mode=='RGB':
        #         transform_list = [transforms.ToTensor(),
        #                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        #     else:
        #         transform_list = [transforms.ToTensor(),
        #                         transforms.Normalize((0.5), (0.5))]

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
            
        A_imgs = []

        A_img = self.transform(A)
        c, w, h = A_img.size()
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
        print(A.shape)
        
        mask = A.clone().zero_()
        
        # let B directly equals A
        B = A.clone()

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetSliding'
