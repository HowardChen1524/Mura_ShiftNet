import pandas as pd
import os
import shutil
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob

parser = parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-mv', '--model_version', type=str, default=None, required=True)
parser.add_argument('-id', '--img_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

args = parser.parse_args()
dataset_version = args.dataset_version
model_version = args.model_version
img_dir = os.path.join(args.img_dir, dataset_version)
save_dir = os.path.join(args.save_dir, dataset_version)

for img_fn in os.listdir(img_dir):
    for th in range(135, 139):
        img_path = os.path.join(img_dir, f'{img_fn}/{th}')        
        save_path = os.path.join(save_dir, img_fn)
        os.makedirs(save_path, exist_ok=True)
        patches = [Image.open(os.path.join(img_path, f'en_{i}.png')) for i in range(0, 225)]
        # patches = [Image.open(os.path.join(img_path, f'en_{i}.png')) for i in range(0, 64)]

        plt.figure(num=img_fn, figsize=(50,50))
        for i in range(1,226):
            plt.subplot(15,15,i)
            plt.axis('off')
            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.imshow(patches[i-1], cmap='gray')
        # for i in range(1,65):
        #     plt.subplot(8,8,i)
        #     plt.axis('off')
        #     plt.subplots_adjust(wspace=0.1, hspace=0.1)
        #     plt.imshow(patches[i-1], cmap='gray')
        plt.savefig(os.path.join(save_path, f'{th}_{img_fn}'))
        plt.clf()
        