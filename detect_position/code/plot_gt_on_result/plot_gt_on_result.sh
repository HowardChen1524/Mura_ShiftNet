#!/bin/bash
dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typed/16/union/0.0150_diff_pos_area_2'

csv_path='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typed/typed.csv'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'

python plot_gt_on_result.py \
-dv=$dataset_version \
-cp=$csv_path \
-dd=$data_dir \
-sd=$save_dir