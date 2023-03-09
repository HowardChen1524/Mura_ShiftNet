#!/bin/bash

dataset_version='typec+b1'
base_dir="/home/sallylab/Howard/Mura_ShiftNet/detect_position"
crop_stride=16

declare th_list=(0.0125 0.015)
declare min_area_list=(0 10 20 22 30 40 50 56 60)
    
for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        data_dir="${base_dir}/${dataset_version}/${crop_stride}/union/${th}_diff_pos_area_${min_area}/imgs"
        gt_dir="${base_dir}/${dataset_version}/actual_pos/ground_truth"
        save_dir="${base_dir}/${dataset_version}/${crop_stride}/union/${th}_diff_pos_area_${min_area}"
        python3 calculate_metrics.py \
        -dv=$dataset_version \
        -dd=$data_dir \
        -gd=$gt_dir \
        -sd=$save_dir
    done
done