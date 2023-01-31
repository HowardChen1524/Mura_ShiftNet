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
img_dir = args.img_dir
save_dir = args.save_dir

for img_fn in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_fn)
    patch_dir = os.path.join(img_path, 'diff')
    save_path = os.path.join(save_dir, f'{dataset_version}/{img_fn}/{th}')
    os.makedirs(save_path, exist_ok=True)
    
    for idx, patch_fn in enumerate(os.listdir(patch_dir)):
        if 'en' in patch_fn: 
            patch_path = os.path.join(patch_dir, patch_fn)
            patch = cv2.imread(patch_path, 0)
            _, res = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY)
            print(res)
            raise
            # cv2.imwrite(os.path.join(save_path, patch_fn), res)
        # else:
        #     patch_path = os.path.join(patch_dir, patch_fn)
        #     patch = cv2.imread(patch_path, 0)
        #     _, res = cv2.threshold(patch, 127, 255, cv2.THRESH_BINARY)
        #     print(res)
        #     raise
        #     # cv2.imwrite(os.path.join(save_path, patch_fn), res)
# for img_fn in os.listdir(img_dir):
#     for th in range(135, 139):
#         img_path = os.path.join(img_dir, img_fn)
#         patch_dir = os.path.join(img_path, 'diff')
#         save_path = os.path.join(save_dir, f'{dataset_version}/{img_fn}/{th}')
#         os.makedirs(save_path, exist_ok=True)
        
#         for idx, patch_fn in enumerate(os.listdir(patch_dir)):
#             if 'en' in patch_fn: 
#                 patch_path = os.path.join(patch_dir, patch_fn)
#                 patch = cv2.imread(patch_path, 0)
#                 _, res = cv2.threshold(patch, th, 255, cv2.THRESH_BINARY)
#                 cv2.imwrite(os.path.join(save_path, patch_fn), res)