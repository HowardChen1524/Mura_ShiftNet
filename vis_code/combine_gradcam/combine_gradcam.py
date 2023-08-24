import os
from collections import defaultdict
from PIL import Image
import numpy as np
import pandas as pd
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('-upd', '--unsup_dir', type=str, default=None, required=True)
parser.add_argument('-spd', '--sup_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

def join_path(p1,p2):
    return os.path.join(p1,p2)

def intersection_two_img(img1, img2):
    
    return (sup_img * unsup_img)*255

if __name__ == '__main__':
    args = parser.parse_args()
    unsup_dir = args.unsup_dir
    sup_dir = args.sup_dir
    save_dir = args.save_dir
    save_dir = join_path(save_dir, 'img')
    os.makedirs(save_dir, exist_ok=True)

    row_data = defaultdict(float)
    dice_list = []
    for fn in os.listdir(unsup_dir):
        print(fn)
        unsup_img = Image.open(join_path(unsup_dir,fn))
        sup_img = Image.open(join_path(sup_dir,fn))
        thresh = 127
        unsup_img = cv2.imread(join_path(unsup_dir,fn), cv2.IMREAD_GRAYSCALE)
        sup_img = cv2.imread(join_path(sup_dir,fn), cv2.IMREAD_GRAYSCALE)
        unsup_img = cv2.threshold(unsup_img, thresh, 255, cv2.THRESH_BINARY)[1]/255
        sup_img = cv2.threshold(sup_img, thresh, 255, cv2.THRESH_BINARY)[1]/255
        
        combine_img = (sup_img * unsup_img)*255
        
        combine_img = Image.fromarray(combine_img).convert('L')
        
        combine_img.save(join_path(save_dir, fn))
    

