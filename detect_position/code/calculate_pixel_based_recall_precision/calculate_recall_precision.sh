#!/bin/bash

dataset_version='typec+b1'
# data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/sup_0.1_unsup_0.0150_56_combined'
# data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/sup_gradcam/SEResNeXt101_d23/0.5'
data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/16/union/0.0125_diff_pos_area_56/imgs'
gt_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/actual_pos/ground_truth'
python calculate_recall_precision.py \
-dv=$dataset_version \
-dd=$data_dir \
-gd=$gt_dir
