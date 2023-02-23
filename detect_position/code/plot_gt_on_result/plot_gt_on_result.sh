#!/bin/bash
dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/sup_0.1_unsup_0.0125_60_combined'
# data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/sup_gradcam/SEResNeXt101_d23/0.1'
# data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/16/union/0.0125_diff_pos_area_60'

csv_path='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/typec+b1.csv'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'

python plot_gt_on_result.py \
-dv=$dataset_version \
-cp=$csv_path \
-dd=$data_dir \
-sd=$save_dir