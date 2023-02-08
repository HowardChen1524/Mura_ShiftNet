#!/bin/bash

dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/detect_position/'
crop_stride=32
declare th_list=(0.0125 0.015)
declare min_area_list=(4 6 8 10 12)

for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        python plot_gt_res.py \
        -dv=$dataset_version \
        -dd=$data_dir \
        -cs=$crop_stride \
        -th=$th
    done
done