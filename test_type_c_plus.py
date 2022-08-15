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
import cv2

def draw_mura_position(fp, fn_series_list, crop_pos_list, stride):
    # 讀取大圖
    img = Image.open(fp)
    
    # =====actual mura===== 
    actual_pos_list = []
    for i in range(0, fn_series_list.shape[0]):
        fn_series = fn_series_list.iloc[i]
        actual_pos_list.append((fn_series['x0'], fn_series['y0'], fn_series['x1'], fn_series['y1']))
        # actual_pos_list.append((fn_series['x0']//3.75, fn_series['y0']//2, fn_series['x1']// 3.75, fn_series['y1']//2))

    for actual_pos in actual_pos_list:
        draw = ImageDraw.Draw(img)  
        draw.rectangle(actual_pos, outline ="yellow")
    
    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    
    # =====predict mura=====
    bounding_box = (64,64)

    for crop_pos in crop_pos_list:
        # max_crop_img_pos 0~255
        x = crop_pos % 15 # 0~15
        y = crop_pos // 15
        # 0,32,64,96,128,160,192,224,256
        crop_x = x * stride
        crop_y = y * stride
        
        # 如果是最後一塊
        if crop_x + bounding_box[0] > 512:
            crop_x = 512 - bounding_box[0] # 448
        if crop_y + bounding_box[1] > 512:
            crop_y = 512 - bounding_box[1] # 448

        pred_pos = [crop_x, crop_y, crop_x+bounding_box[0]-1, crop_y+bounding_box[1]-1]

        # create rectangle image
        draw = ImageDraw.Draw(img)  
        draw.rectangle(pred_pos, outline ="red")

    img.save(f"./type_c_plus_smura_pos_img/{fn_series_list.iloc[0]['fn'][:-4]}_mura_pos.png")


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
    print(df.iloc[(df['h']+df['w']).argmax()])
    print(df.iloc[(df['h']+df['w']).argmin()])

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

        fp = data['A_paths'][0]
        fn = fp[len(opt.testing_smura_dataroot):]
        fn_series_list = df[df['fn']==fn]
        top_n = fn_series_list.shape[0]
        
        crop_pos_list = np.argsort(-crop_scores)[:top_n] # 取前 n 張
        
        # 畫圖 & 新增結果到 overlapping_df
        draw_mura_position(fp, fn_series_list, crop_pos_list, opt.crop_stride)
    
        # append 每張小圖的 MSE
        if i == 0:
            all_crop_scores = crop_scores.copy()
        else:
            all_crop_scores = np.append(all_crop_scores, crop_scores)
    
    # 計算所有圖片平均
    s_mean = all_crop_scores.mean()
    s_std = all_crop_scores.std()

    print(f"Smura mean: {s_mean}")
    print(f"Smura std: {s_std}")


