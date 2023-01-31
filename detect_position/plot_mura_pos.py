import os
import argparse
import shutil
from collections import defaultdict 
import xmltodict, json
from glob import glob
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

args = parser.parse_args()
dataset_version = args.dataset_version
data_dir = args.data_dir
save_dir = args.save_dir

def add_row(info_fn, fn, obj):
    info_fn['fn'] = fn
    info_fn['smura_name'] = obj['name']
    x0 = int(obj['bndbox']['xmin'])
    info_fn['x0'] = x0
    y0 = int(obj['bndbox']['ymin'])
    info_fn['y0'] = y0
    x1 = int(obj['bndbox']['xmax'])
    info_fn['x1'] = x1
    y1 = int(obj['bndbox']['ymax'])
    info_fn['y1'] = y1
    info_fn['x_center'] = (x0+x1)//2
    info_fn['y_center'] = (y0+y1)//2
    # info_fn['w'] = x1-x0+1
    # info_fn['h'] = y1-y0+1
    info_fn['w'] = x1-x0
    info_fn['h'] = y1-y0
    return info_fn

# 將所有標註 xml 統整成標註 df
xml_dir = os.path.join(data_dir, f'{dataset_version}/xml')
xml_list = glob(f"{os.path.join(xml_dir, '*xml')}")
info_fn_list = []
for xml_path in xml_list:
    with open(xml_path) as fd:
        json_fd = xmltodict.parse(fd.read())
        if isinstance(json_fd['annotation']['object'], list):
        # if type(json_fd['annotation']['object']) == list:
            for obj in json_fd['annotation']['object']:
                info_fn = defaultdict()
                # print(info_f)
                # print(json_fd['annotation']['filename'])
                # print(obj)
                info_fn_list.append(add_row(info_fn, json_fd['annotation']['filename'], obj))
        else:
            info_fn = defaultdict()
            info_fn_list.append(add_row(info_fn, json_fd['annotation']['filename'], json_fd['annotation']['object']))

df = pd.DataFrame.from_dict(info_fn_list)
# df.to_csv(os.path.join(save_path, f'{dataset_version}.csv'), index=False, header=True)

# 基於標註 df 將實際 mura 位置標註在圖上
save_dir = os.path.join(save_dir, f'{dataset_version}/actual_pos')
os.makedirs(save_dir, exist_ok=True)
img_dir = os.path.join(data_dir, f'{dataset_version}/img')
img_list = glob(f"{os.path.join(img_dir, '*png')}")
for img_path in img_list:
    fn = img_path.split('/')[-1]
    img = Image.open(img_path)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fn_series_list = df[df['fn']==fn]
    
    actual_pos_list = []
    for i in range(0, fn_series_list.shape[0]):
        fn_series = fn_series_list.iloc[i]
        actual_pos_list.append((int(fn_series['x0']/3.75), int(fn_series['y0']/2.109375), int(fn_series['x1']/ 3.75), int(fn_series['y1']/2.109375)))

    for actual_pos in actual_pos_list:
        draw = ImageDraw.Draw(img)  
        draw.rectangle(actual_pos, outline ="yellow")
    # print(os.path.join(save_dir, fn))
    img.save(os.path.join(save_dir, fn))
