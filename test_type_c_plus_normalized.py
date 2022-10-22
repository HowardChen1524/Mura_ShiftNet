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

def draw_mura_position(isMask, save_path, fp, fn_series_list, crop_pos_list, stride):
    # 讀取大圖
    img = Image.open(fp)
    img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    # =====actual mura===== 
    actual_pos_list = []
    for i in range(0, fn_series_list.shape[0]):
        fn_series = fn_series_list.iloc[i]
        # actual_pos_list.append((fn_series['x0'], fn_series['y0'], fn_series['x1'], fn_series['y1']))
        actual_pos_list.append((int(fn_series['x0']/3.75), int(fn_series['y0']/2.109375), int(fn_series['x1']/ 3.75), int(fn_series['y1']/2.109375)))
    
    for actual_pos in actual_pos_list:
        draw = ImageDraw.Draw(img)  
        draw.rectangle(actual_pos, outline ="yellow")
    
    # =====predict mura=====
    if isMask:
        bounding_box = (32,32)
    else:
        bounding_box = (64,64)
    for crop_pos in crop_pos_list:
        x = crop_pos % 15 # 0~14
        y = crop_pos // 15
        # 0,32,64,96,128,160,192,224,256...448
        crop_x = x * stride
        crop_y = y * stride
        
        # 如果是最後一塊
        if crop_x + bounding_box[0] > 512:
            crop_x = 512 - bounding_box[0] # 448
        if crop_y + bounding_box[1] > 512:
            crop_y = 512 - bounding_box[1] # 448

        if isMask:
          pred_pos = [crop_x+(bounding_box[0]//2), crop_y+(bounding_box[1]//2), crop_x+(bounding_box[0]//2)+bounding_box[0]-1, crop_y+(bounding_box[1]//2)+bounding_box[1]-1]
        else:
          pred_pos = [crop_x, crop_y, crop_x+bounding_box[0]-1, crop_y+bounding_box[1]-1]

        # create rectangle image
        draw = ImageDraw.Draw(img)  
        draw.rectangle(pred_pos, outline ="red")

    img.save(f"{save_path}position/{fn_series_list.iloc[0]['fn'][:-4]}_draw_smura_position.png")
    img.save(f"{save_path}check_inpaint_img/{fn_series_list.iloc[0]['fn'][:-4]}/draw_smura_position.png")


if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    opt.serial_batches = False
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!
    save_path = f"./{opt.inpainting_mode}_{opt.measure_mode}/"
    mkdir(save_path)
    data_loader = CreateDataLoader(opt)
    dataset = defaultdict()
    dataset['normal'] = data_loader['normal'].load_data()
    dataset['smura'] = data_loader['smura'].load_data()
    
    
    model = create_model(opt)

    n_all_crop_scores = []
    s_all_crop_scores = None

    for i, data in enumerate(dataset['normal']):
        print(f"img: {i}")
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)

        model.set_input(data) 
        crop_scores = model.test() # 225 張小圖的 score
    
        n_all_crop_scores.append(crop_scores)
        
    n_all_crop_scores = np.array(n_all_crop_scores)
    print(n_all_crop_scores.shape)
    n_pos_mean = np.mean(n_all_crop_scores, axis=0)
    n_pos_std = np.std(n_all_crop_scores, axis=0)
    
    df = pd.read_csv('./Mura_type_c_plus.csv')
    print(df.iloc[(df['h']+df['w']).argmax()][['fn','w','h']])
    print(df.iloc[(df['h']+df['w']).argmin()][['fn','w','h']])
    mkdir(f"{save_path}check_inpaint_img/")
    mkdir(f"{save_path}position/")
    for i, data in enumerate(dataset['smura']):
        print(f"img: {i}")
        # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)

        # 建立 input real_A & real_B
        # it not only sets the input data with mask, but also sets the latent mask.

        fp = data['A_paths'][0]
        fn = fp[len(opt.testing_smura_dataroot):]
        fn_series_list = df[df['fn']==fn]
        mkdir(f'{save_path}check_inpaint_img/{fn[:-4]}/')

        model.set_input(data) 
        crop_scores = model.test(fn) # 225 張小圖的 score
        for pos in range(0,crop_scores.shape[0]):
            crop_scores[pos] = (crop_scores[pos]-n_pos_mean[pos])/n_pos_std[pos]
        
        # print(crop_scores)
        top_n = fn_series_list.shape[0]
        crop_pos_list = np.argsort(-crop_scores)[:top_n] # 取前 n 張
        
        crop_pos_list_str = [f"{pos}\n" for pos in crop_pos_list]
        with open(f"{save_path}check_inpaint_img/{fn[:-4]}/predicted_position.txt", 'w') as f:
            f.writelines(crop_pos_list_str)

        # 畫圖 & 新增結果到 overlapping_df
        if "Mask" in opt.measure_mode or "Discount" in opt.measure_mode:
            draw_mura_position(True, save_path, fp, fn_series_list, crop_pos_list, opt.crop_stride)
        else:
            draw_mura_position(False, save_path, fp, fn_series_list, crop_pos_list, opt.crop_stride)
        # append 每張小圖的 MSE
        if i == 0:
            s_all_crop_scores = crop_scores.copy()
        else:
            s_all_crop_scores = np.append(s_all_crop_scores, crop_scores)
    
    # 計算所有圖片平均
    n_mean = n_all_crop_scores.mean()
    n_std = n_all_crop_scores.std()
    s_mean = s_all_crop_scores.mean()
    s_std = s_all_crop_scores.std()

    print(f"Normal mean: {n_mean}")
    print(f"Normal std: {n_std}")
    print(f"Smura mean: {s_mean}")
    print(f"Smura std: {s_std}")


