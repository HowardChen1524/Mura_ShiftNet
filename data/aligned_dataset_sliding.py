#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

general_size = (471,281)
stride = 256

class AlignedDatasetSliding(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = opt.dataroot # More Flexible for users

        self.A_paths = sorted(make_dataset(self.dir_A)) # image path list

        assert(opt.resize_or_crop == 'resize_and_crop')

        if opt.isTrain:
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5)),
                            transforms.RandomCrop(self.opt.fineSize)]


            self.transform = transforms.Compose(transform_list)
        else:
            transform_list = [transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
            self.transform = transforms.Compose(transform_list)                                    

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')
        # print(A.size)
        # resize general size 471*281
        A = A.resize(general_size, Image.BICUBIC)
        # print(A.size)

        A_imgs = []
        if self.opt.isTrain:
            crop_num = 4 # be param
            for i in range(crop_num):
                A_imgs.append(self.transform(A))
                # print(A_imgs[i].shape)
                # print(A_imgs[i])
        else:
            A_img = self.transform(A)
            w, h = A.size # w = 471 h = 281
            y_end_crop, x_end_crop = False, False
            for y in range(0, w, stride):
                y_end_crop = False
                for x in range(0, h, stride):
                    x_end_crop = False
                    crop_y = y
                    if (y + self.opt.fineSize) > w:
                        crop_y =  w - self.opt.fineSize
                        y_end_crop = True

                    crop_x = x
                    if (x + self.opt.fineSize) > h:
                        crop_x = h - self.opt.fineSize
                        x_end_crop = True
                    # print(f"crop_y: {crop_y}")
                    # print(f"crop_x: {crop_x}")
                    img = transforms.functional.crop(A_img, crop_y, crop_x, self.opt.fineSize, self.opt.fineSize)
                    A_imgs.append(img)
                    if x_end_crop:
                        break
                if x_end_crop and y_end_crop:
                    break
        
        A = torch.stack(A_imgs)
        # print(A.shape)

        # print('A_End')
        #if (not self.opt.no_flip) and random.random() < 0.5:
        #    idx = [i for i in range(A.size(2) - 1, -1, -1)] # size(2)-1, size(2)-2, ... , 0
        #    idx = torch.LongTensor(idx)
        #    A = A.index_select(2, idx)

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A[0].clone().zero_()
        
        # let B directly equals A
        B = A.clone()
        # print(B.shape)
        # print('B_End')

        return {'A': A, 'B': B, 'M': mask, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetSliding'
