#!/bin/bash
dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/4-connected/sup_0.1_unsup_0.0125_60_combined'

csv_path='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/typec+b1.csv'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'

python plot_gt_on_result.py \
-dv=$dataset_version \
-cp=$csv_path \
-dd=$data_dir \
-sd=$save_dir