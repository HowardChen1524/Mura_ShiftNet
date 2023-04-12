import os
import argparse
from collections import defaultdict 
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import sys 


parser = argparse.ArgumentParser()
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-gd', '--gt_dir', type=str, default=None, required=True)
parser.add_argument('-cp', '--csv_path', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)
parser.add_argument('-rs', '--resized', type=int, default=None, required=True)

def join_path(p1,p2):
    return os.path.join(p1,p2)

if __name__ == '__main__': 
    args = parser.parse_args()
    data_dir = args.data_dir
    gt_dir = args.gt_dir
    csv_path = args.csv_path
    save_dir = args.save_dir
    isResize = args.resized

    os.makedirs(save_dir, exist_ok=True)


    df = pd.read_csv(csv_path)
    # 基於標註 df 將實際 mura 位置標註在圖上
    img_list = glob(f"{join_path(data_dir, '*png')}")
    gt_list = [i for i in os.listdir(gt_dir)]
    print(gt_list)
    for img_path in img_list:
        fn = img_path.split('/')[-1]
        print(fn)
        if fn not in gt_list:
            continue
        print(fn)
        img = Image.open(img_path)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fn_series_list = df[df['fn']==fn]
        
        actual_pos_list = []
        for i in range(0, fn_series_list.shape[0]):
            fn_series = fn_series_list.iloc[i]
            if isResize:
                actual_pos_list.append((int(fn_series['x0']/3.75), int(fn_series['y0']/2.109375), int(fn_series['x1']/ 3.75), int(fn_series['y1']/2.109375)))
            else:
                actual_pos_list.append((int(fn_series['x0']), int(fn_series['y0']), int(fn_series['x1']), int(fn_series['y1'])))

        for actual_pos in actual_pos_list:
            draw = ImageDraw.Draw(img)  
            draw.rectangle(actual_pos, outline ="red")
            # print(actual_pos)

        img.save(join_path(save_dir, fn))