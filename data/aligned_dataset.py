#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = opt.dataroot # dataset 位置
        self.A_paths = sorted(make_dataset(self.dir_A)) # make_dataset 回傳 image 的 filename path list
        if self.opt.offline_loading_mask: # 預設 False, 用自己的 mask
            self.mask_folder = self.opt.training_mask_folder if self.opt.isTrain else self.opt.testing_mask_folder
            self.mask_paths = sorted(make_dataset(self.mask_folder))

        assert(opt.resize_or_crop == 'resize_and_crop')

        # 前處理部分
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        # transform_list = [transforms.ToTensor(),
        #                   transforms.Normalize((0.5),
        #                                        (0.5))]
        self.transform = transforms.Compose(transform_list)
        
    def __getitem__(self, index):
        # read image
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        # A = Image.open(A_path).convert('L')
        
        w, h = A.size
        # print(f"w: {w}") # 1920
        # print(f"h: {h}") # 1080
        
        # ori crop
        if w < h:
            ht_1 = self.opt.loadSize * h // w
            wd_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)
        else:
            wd_1 = self.opt.loadSize * w // h
            ht_1 = self.opt.loadSize
            print(f"wd_1w: {wd_1}") # 455
            A = A.resize((wd_1, ht_1), Image.BICUBIC)
        # 進行前處理
        A = self.transform(A)
        h = A.size(1)
        w = A.size(2)

        # crop image
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        # print(f"w_offset: {w_offset}")
        # print(f"h_offset: {h_offset}")
        A = A[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize] # color, row, col 

        if (not self.opt.no_flip) and random.random() < 0.5: # self.opt.no_flip 預設 False
            A = torch.flip(A, [2]) # torch.flip(input, dims) → Tensor Reverse the order of a n-D tensor along given axis in dims.
        
        # let B directly equals to A
        B = A.clone()
        A_flip = torch.flip(A, [2])
        B_flip = A_flip.clone()

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A.clone().zero_()
        if self.opt.offline_loading_mask:
            if self.opt.isTrain:
                mask = Image.open(self.mask_paths[random.randint(0, len(self.mask_paths)-1)])
            else:
                mask = Image.open(self.mask_paths[index % len(self.mask_paths)])
            mask = mask.resize((self.opt.fineSize, self.opt.fineSize), Image.NEAREST)
            mask = transforms.ToTensor()(mask)
    
        # 用 dict 回傳
        return {'A': A, 'B': B, 'A_F': A_flip, 'B_F': B_flip, 'M': mask,
                'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
