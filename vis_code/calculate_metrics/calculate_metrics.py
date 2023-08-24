import os
from collections import defaultdict
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import confusion_matrix
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-gd', '--gt_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)
parser.add_argument('-ir', '--isResize', type=int, default=None, required=True)

def compute_recall_precision(y_true, y_pred):
    assert y_true.shape == y_pred.shape, "Error: Images have different shapes"
    tn, fp, fn, tp = confusion_matrix(y_true.ravel(), y_pred.ravel()).ravel()
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    return recall, precision

def dice_coefficient(img1, img2):    
    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "Error: Images have different shapes"
    # Calculate the Dice coefficient
    # Calculate the intersection
    intersection = np.sum(img1 * img2)
    total_white_pixel = np.sum(img1) + np.sum(img2)

    dice = (2 * intersection) / total_white_pixel
    return dice

def join_path(p1,p2):
    return os.path.join(p1,p2)

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    gt_dir = args.gt_dir
    save_dir = args.save_dir
    isResize = args.isResize
    os.makedirs(save_dir, exist_ok=True)
    
    dice_mean = defaultdict(float)
    recall_mean = defaultdict(float)
    precision_mean = defaultdict(float)
    count = 0
    for fn in os.listdir(gt_dir):
        count += 1   
        # Load the images
        thresh = 127
        gt_img = cv2.imread(join_path(gt_dir,fn), cv2.IMREAD_GRAYSCALE)
        diff_img = cv2.imread(join_path(data_dir,fn), cv2.IMREAD_GRAYSCALE)
        if isResize == 1:
            gt_img = cv2.resize(gt_img, (512,512), interpolation=cv2.INTER_LINEAR)
            diff_img = cv2.resize(diff_img, (512,512), interpolation=cv2.INTER_LINEAR)
        gt_img = cv2.threshold(gt_img, thresh, 255, cv2.THRESH_BINARY)[1]/255
        diff_img = cv2.threshold(diff_img, thresh, 255, cv2.THRESH_BINARY)[1]/255

        # print(gt_img.shape)
        # print(diff_img.shape)
        # print(np.unique(gt_img))
        # print(np.unique(diff_img))
        # raise
        dice = dice_coefficient(gt_img, diff_img)
        dice_mean[fn] = dice            
        recall, precision = compute_recall_precision(gt_img, diff_img)
        recall_mean[fn] = recall
        precision_mean[fn] = precision
        print("Num {}: {}\ndice: {}, recall: {}, precision:{}".format(count, fn, dice, recall, precision))
        
    df_dice = pd.DataFrame(data=list(dice_mean.items()),columns=['fn','dice'])
    df_recall = pd.DataFrame(data=list(recall_mean.items()),columns=['fn','recall'])
    df_precision = pd.DataFrame(data=list(precision_mean.items()),columns=['fn','precision'])

    print(f"finished, dice mean:{df_dice['dice'].mean()}")
    print(f"finished, recall mean:{df_recall['recall'].mean()}")
    print(f"finished, precision mean:{df_precision['precision'].mean()}")
    df_dice.to_csv(join_path(save_dir, f'dice_all.csv'),index=False)
    df_recall.to_csv(join_path(save_dir, f'recall_all.csv'),index=False)
    df_precision.to_csv(join_path(save_dir, f'precision_all.csv'),index=False)

    with open(join_path(save_dir, f"result_all.txt"), 'w') as f:
        msg = f"All img: {count}\n" 
        msg += f"hit num: {df_dice[df_dice['dice']>0].shape[0]}\n"
        msg += f"dice mean: {df_dice['dice'].mean()}\n"
        msg += f"recall mean: {df_recall['recall'].mean()}\n"
        msg += f"precision mean: {df_precision['precision'].mean()}\n"
        f.writelines(msg)