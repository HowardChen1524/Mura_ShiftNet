import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

import torch
from PIL import Image, ImageDraw
import numpy as np 
import pandas as pd
from util.util import tensor2im, mkdir
from collections import defaultdict

def draw_mura_position(df, fp, dir_plen, max_crop_img_pos, stride):
    # 讀取大圖
    print(fp)
    h,w = 512,512
    img = Image.open(fp).convert('RGB')
    print(img.size)
    img = img.resize((h, w), Image.BICUBIC)
    print(img.size)
    img_std = img.copy()

    # 宣告變數
    actual_pos = []
    pred_pos = []

    # =====actual mura=====  
    # bounding_box 54*100
    bounding_box_x = int(200 / 3.75) + 1 # +1 是為了使邊長為偶數
    bounding_box_y = int(200 / 2)

    # 擷取檔名並將副檔名轉成bmp，方便比較 MURA_XY.csv
    fn = fp[dir_plen:].replace('png','bmp')
    print(fn)
    fn_series = df[df['PIC_ID'] == fn]
    print(fn_series)

    # 結果變數宣告
    res = {'PIC_ID': fn, 'Actual_pos': str(actual_pos), 'Actual_std': [np.nan],'Predict_pos': str(pred_pos), 'Predict_std': [np.nan], 'Overlapping': [-1], 'std_actual_larger_predict': [-1]}

    # 如果 MURA_XY.csv 裡面xy座標有空值，直接 return
    if fn_series.isna()['X'].values[0] or fn_series.isna()['Y'].values[0]:
        return pd.DataFrame(res)
 
    crop_x0,crop_y0,crop_x1,crop_y1 = 0,0,0,0
    if fn_series['PRODUCT_CODE'].values[0] == 'T850QVN03': # 4k
        crop_x0 = int(fn_series['X'] / 2 / 3.75) - (bounding_box_x // 2)
        crop_y0 = int(fn_series['Y'] / 2 / 2) - (bounding_box_y // 2)
        crop_x1 = int(fn_series['X'] / 2 / 3.75) + (bounding_box_x // 2)
        crop_y1 = int(fn_series['Y'] / 2 / 2) + (bounding_box_y // 2)
    elif fn_series['PRODUCT_CODE'].values[0] == 'T850MVR05': # 8k
        crop_x0 = int(fn_series['X'] / 4 / 3.75) - (bounding_box_x // 2)
        crop_y0 = int(fn_series['Y'] / 4 / 2) - (bounding_box_y // 2)
        crop_x1 = int(fn_series['X'] / 4 / 3.75) + (bounding_box_x // 2)
        crop_y1 = int(fn_series['Y'] / 4 / 2) + (bounding_box_y // 2)
    
    actual_pos = [crop_x0, crop_y0, crop_x1, crop_y1]
    for pos in range(len(actual_pos)):
        if actual_pos[pos] < 0:
            actual_pos[pos] = 0
        if actual_pos[pos] > 511:
            actual_pos[pos] = 511

    draw = ImageDraw.Draw(img)  
    draw.rectangle(actual_pos, outline ="yellow")

    # =====predict mura=====
    # create bounding_box 64*64
    bounding_box_x = 64
    bounding_box_y = 64

    # max_crop_img_pos 0~255
    x = max_crop_img_pos % 16 # 0~15
    y = max_crop_img_pos // 16

    crop_x = x * opt.crop_stride -1
    crop_y = y * opt.crop_stride -1
    
    # 如果是第一塊 crop_x, crop_y 會是 -1
    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0

    # 如果是最後一塊
    if x == 15:
        crop_x = w -bounding_box_x -1 # 447
    if y == 15:
        crop_y = h -bounding_box_y -1 # 447

    pred_pos = [crop_x, crop_y, crop_x+bounding_box_x, crop_y+bounding_box_y]

    # create rectangle image
    draw = ImageDraw.Draw(img)  
    draw.rectangle(pred_pos, outline ="red")
    img.save(f'./mura_pos_img/{fp[dir_plen:-4]}_mura_pos.png')

    # 沒有重疊
    img_std = np.array(img_std)
    actual_std = img_std[actual_pos[0]:actual_pos[2], actual_pos[1]:actual_pos[3]].std()
    predict_std = img_std[pred_pos[0]:pred_pos[2], pred_pos[1]:pred_pos[3]].std()
    res['Actual_pos'] = str(actual_pos)
    res['Actual_std'] = [actual_std]
    res['Predict_pos'] = str(pred_pos)
    res['Predict_std'] = [predict_std]   
    
    if (actual_pos[0]>=pred_pos[2]) or (actual_pos[2]<=pred_pos[0]) or (actual_pos[3]<=pred_pos[1]) or (actual_pos[1]>=pred_pos[3]):
        res['Overlapping'] = [0]
    else:
        res['Overlapping'] = [1]
    
    if actual_std >= predict_std:
        res['std_actual_larger_predict'] = [1]
    else:
        res['std_actual_larger_predict'] = [0]

    return pd.DataFrame(res)

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    opt.serial_batches = False
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    data_loader = CreateDataLoader(opt)
    dataset = data_loader['smura'].load_data()
    
    model = create_model(opt)

    s_score_log = []
    all_crop_scores = []
    
    mkdir('./type_c_plus_smura_pos_img/')
    df = pd.read_csv('./Mura_type_c_plus.csv')
    overlapping_df = pd.DataFrame()

    for i, data in enumerate(dataset):
        # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)

        # 建立 input real_A & real_B
        # it not only sets the input data with mask, but also sets the latent mask.

        model.set_input(data) 
        crop_scores = model.test() # 256 張小圖的 score

        fn = data['A_paths'][0][len(opt.testing_smura_dataroot):]
        fn_series = df[df['fn']==fn]
        print(fn_series)
        raise
        # 取前 n 張
        max_crop_score = np.max(crop_scores)
        print(max_crop_score)
        max_crop_img_pos = np.argmax(crop_scores)

        # append 每張小圖的 MSE
        if i == 0:
            all_crop_scores = crop_scores.copy()
        else:
            all_crop_scores = np.append(all_crop_scores, crop_scores)

        # 畫圖 & 新增結果到 overlapping_df
        res = draw_mura_position(df, data['A_paths'][0], dir_path_len, max_crop_img_pos, opt.crop_stride)
        # print(res)
        overlapping_df = pd.concat([overlapping_df, res],ignore_index=True)

    all_overlapping = overlapping_df['Overlapping'].to_numpy()
    all_overlapping = all_overlapping[all_overlapping!=-1] # 去掉缺失值
    all_std = overlapping_df['std_actual_larger_predict'].to_numpy()
    all_std = all_std[all_std!=-1] # 去掉缺失值

    overlap_ratio = all_overlapping[np.where(all_overlapping==1)].shape[0]/all_overlapping.shape[0]
    print(f"overlap_ratio: {all_overlapping[np.where(all_overlapping==1)].shape[0]}/{all_overlapping.shape[0]} , {overlap_ratio}")
    std_ratio = all_std[np.where(all_std==1)].shape[0]/all_std.shape[0]
    print(f"std_actual_larger_predict_ratio: {all_std[np.where(all_std==1)].shape[0]}/{all_std.shape[0]}, {std_ratio}")
    overlapping_df.to_csv('./overlap_res.csv')
    
    # 計算所有圖片平均
    s_mean = all_crop_scores.mean()
    s_std = all_crop_scores.std()

    print(f"Smura mean: {s_mean}")
    print(f"Smura std: {s_std}")


