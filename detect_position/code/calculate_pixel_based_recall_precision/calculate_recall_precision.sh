#!/bin/bash

declare th_list=(0.0125 0.0150)
declare min_area_list=(0 10 20 22 30 40 50 56 60)

dataset_version='typec+b1'
gt_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/actual_pos/ground_truth'
for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        data_dir="/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/4-connected/sup_0.5_unsup_${th}_${min_area}_combined"
        # data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/sup_gradcam/SEResNeXt101_d23/0.5'
        # data_dir="/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/8-connected/16/union/${th}_diff_pos_area_${min_area}"
        
        python calculate_recall_precision.py \
        -dv=$dataset_version \
        -dd=$data_dir \
        -gd=$gt_dir
    done
done