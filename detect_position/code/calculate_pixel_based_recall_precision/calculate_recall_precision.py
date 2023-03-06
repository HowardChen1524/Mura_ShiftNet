import os
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-gd', '--gt_dir', type=str, default=None, required=True)

def compute_recall_precision(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return recall, precision, f1

def join_path(path1, path2):
    return (os.path.join(path1,path2))

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_version = args.dataset_version
    data_dir = args.data_dir
    gt_dir = args.gt_dir

    pixels_imgs = []
    pixels_gt = []
    for fn in os.listdir(gt_dir):
        # Load the images
        img = (np.array(Image.open(join_path(data_dir, f'imgs/{fn}')))/255)
        # img = (np.array(Image.open(join_path(data_dir, fn)))/255)
        gt = (np.array(Image.open(join_path(gt_dir,fn)))/255)
        pixels_gt.append(gt)
        pixels_imgs.append(img)
    
    pixels_gt = np.array(pixels_gt).flatten()
    pixels_imgs = np.array(pixels_imgs).flatten()
    # print(np.unique(pixels_gt))
    # print(np.unique(pixels_imgs))
    # print(pixels_gt[:5])
    # print(pixels_imgs[:5])
    # raise
    recall, precision, f1 = compute_recall_precision(pixels_gt, pixels_imgs)
    with open (join_path(data_dir,f"metrics.txt"), 'w') as f:
        msg  = f"recall: {recall}\n"
        msg += f"precision: {precision}\n"
        msg += f"f1: {f1}\n"
        f.write(msg)
    print('finished')