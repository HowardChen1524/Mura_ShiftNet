import pandas as pd
import os
import shutil
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # like pil image
import argparse
from glob import glob

parser = parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-cs', '--crop_stride', type=str, default=None, required=True)
parser.add_argument('-ths', '--threshold_list', type=str, default=None, required=True)

args = parser.parse_args()
dataset_version = args.dataset_version
data_dir = args.data_dir
crop_stride = args.crop_stride
th_list = args.threshold_list.split(',')

actual_dir = os.path.join(args.data_dir, f'{dataset_version}/actual_pos')
save_dir = os.path.join(args.data_dir, f'{dataset_version}/{crop_stride}/diff_image_compare')
os.makedirs(save_dir, exist_ok=True)

for fn in os.listdir(actual_dir):
    actual_img = mpimg.imread(os.path.join(actual_dir, fn))
    diff_img_list = []
    for idx, th in enumerate(th_list):
        union_dir = os.path.join(args.data_dir, f'{dataset_version}/{crop_stride}/union/{th}_diff_pos')
        diff_img_list.append(mpimg.imread(os.path.join(union_dir, fn)))

    fig = plt.figure(figsize=(15,8))
    plt.subplots_adjust(wspace=0.05)
    plt.subplot(1,4,1)
    plt.title('GroundTruth', fontsize=18)
    plt.axis('off')
    plt.imshow(actual_img)

    plt.subplot(1,4,2)
    plt.title(f'{th_list[0]}', fontsize=18)
    plt.axis('off')
    plt.imshow(diff_img_list[0])

    plt.subplot(1,4,3)
    plt.title(f'{th_list[1]}', fontsize=18)
    plt.axis('off')
    plt.imshow(diff_img_list[1])

    plt.subplot(1,4,4)
    plt.title(f'{th_list[2]}', fontsize=18)
    plt.axis('off')
    plt.imshow(diff_img_list[2])

    plt.savefig(os.path.join(save_dir, fn), transparent=True)
    plt.clf()