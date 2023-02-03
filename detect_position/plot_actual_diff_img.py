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
parser.add_argument('-th', '--threshold', type=str, default=None, required=True)

args = parser.parse_args()
dataset_version = args.dataset_version
data_dir = args.data_dir
crop_stride = args.crop_stride
th = args.threshold

actual_dir = os.path.join(args.data_dir, f'{dataset_version}/actual_pos')
union_dir = os.path.join(args.data_dir, f'{dataset_version}/{crop_stride}/union/{th}_diff_pos')
# mean_dir = os.path.join(args.data_dir, f'{dataset_version}/{crop_stride}/mean/{th}_diff_pos')
save_dir = os.path.join(args.data_dir, f'{dataset_version}/{crop_stride}/')

for fn in os.listdir(actual_dir):
    actual_img = mpimg.imread(os.path.join(actual_dir, fn))
    union_diff_img = mpimg.imread(os.path.join(union_dir, fn))
    # mean_diff_img = mpimg.imread(os.path.join(mean_dir, fn))
    
    save_path = os.path.join(save_dir, f'{th}_actual_union_compare')
    os.makedirs(save_path, exist_ok=True)
    fig = plt.figure(figsize=(15,8))
    plt.subplots_adjust(wspace=0.05)
    plt.subplot(1,2,1)
    plt.title('GroundTruth', fontsize=18)
    plt.axis('off')
    plt.imshow(actual_img)
    plt.subplot(1,2,2)
    plt.title('Shift-Net Union', fontsize=18)
    plt.axis('off')
    plt.imshow(union_diff_img)
    plt.savefig(os.path.join(save_path, fn), transparent=True)
    plt.clf()

    # save_path = os.path.join(save_dir, f'{th}_actual_mean_compare')
    # os.makedirs(save_path, exist_ok=True)
    # fig = plt.figure(figsize=(15,8))
    # plt.subplots_adjust(wspace=0.05)
    # plt.subplot(1,2,1)
    # plt.title('GroundTruth', fontsize=18)
    # plt.axis('off')
    # plt.imshow(actual_img)
    # plt.subplot(1,2,2)
    # plt.title('Shift-Net Mean', fontsize=18)
    # plt.axis('off')
    # plt.imshow(mean_diff_img)
    # plt.savefig(os.path.join(save_dir, fn), transparent=True)
    # plt.clf()

    # save_path = os.path.join(save_dir, 'all_compare')
    # os.makedirs(save_path, exist_ok=True)
    # fig = plt.figure(figsize=(15,8))
    # plt.subplots_adjust(wspace=0.05)
    # plt.subplot(1,3,1)
    # plt.title('GroundTruth', fontsize=18)
    # plt.axis('off')
    # plt.imshow(actual_img)
    # plt.subplot(1,3,2)
    # plt.title('Shift-Net Union', fontsize=18)
    # plt.axis('off')
    # plt.imshow(union_diff_img)
    # plt.subplot(1,3,3)
    # plt.title('Shift-Net Mean', fontsize=18)
    # plt.axis('off')
    # plt.imshow(mean_diff_img)
    # plt.savefig(os.path.join(save_path, fn), transparent=True)
    # plt.clf()