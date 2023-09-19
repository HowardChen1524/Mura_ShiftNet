import time
import os
import glob
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model

from util.utils import mkdir, data_transforms, make_single_dataloader, set_seed

import torchvision.models as models

from data.AI9_dataset import AI9_Dataset
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import glob
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import InterpolationMode
import argparse
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.patches import Rectangle

fast_tran = transforms.Compose([
    transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor()
])

def initail_setting():
  opt = TestOptions().parse()
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batchSize = 1  # test code only supports batchSize = 1
  opt.serial_batches = True  # no shuffle
  set_seed(2022)
  
  return opt, opt.gpu_ids[0]

def join_path(p1, p2):
    return os.path.join(p1,p2)

def supervised_model_gradcam(opt, gpu):
    test_img_files = glob.glob(join_path(opt.testing_smura_dataroot,'*.png'))
    # test_img_files = test_img_files[:5] 
    test_targets = [1]*len(test_img_files)
    test_files = [f.split("/")[-1] for f in test_img_files]

    ds = AI9_Dataset(feature = test_img_files,
                        target = test_targets,
                        name = test_files,
                        transform = data_transforms["test"])
          
    dataloader = make_single_dataloader(ds)
    # read model
    # seresnext101
    model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
    model_sup.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    sup_model_path = os.path.join(opt.checkpoints_dir, f"{opt.sup_model_version}/model.pt")
    print(sup_model_path)
    model_sup.load_state_dict(torch.load(sup_model_path, map_location=torch.device(f"cuda:{gpu}")))  

    # init cam
    cam = GradCAM(model=model_sup, target_layers=[model_sup.layers[-1]], use_cuda=True)

    rgb_images = [Image.open(p) for p in test_img_files]
    grayscale_cam = []
    print(len(dataloader))

    for x, y, n in tqdm(dataloader):
        grayscale_cam.append(cam(input_tensor=x, aug_smooth=False, eigen_smooth=False))
    
    top_k = opt.top_k
    for index, (cam, rgb_img) in enumerate(zip(grayscale_cam, rgb_images)):
        
        name = test_files[index]
        cam_discrete = cam[0] > opt.sup_gradcam_th
        mask = np.ones((256, 256), dtype='uint8')*255
        mask[cam_discrete == False] = 0
            
        if opt.resolution == 'resized':
            RESOLUTION = (512,512)
        else:
            RESOLUTION = (1920,1080)
        mask = Image.fromarray(mask).resize(RESOLUTION, Image.Resampling.BILINEAR).convert('L')
        save_path = os.path.join(opt.results_dir, 'img')
        mkdir(save_path)
        mask.save(join_path(save_path, name))

if __name__ == '__main__':
  
    opt, gpu = initail_setting()  

    # ===== supervised =====
    res_sup = supervised_model_gradcam(opt, gpu)
