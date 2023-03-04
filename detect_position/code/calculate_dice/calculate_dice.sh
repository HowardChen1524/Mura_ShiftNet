#!/bin/bash

dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'
crop_stride=16
declare th_list=(0.0125 0.015)
declare min_area_list=(0 10 20 22 30 40 50 56 60)

for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        python calculate_dice.py \
        -dv=$dataset_version \
        -dd=$data_dir \
        -cs=$crop_stride \
        -th=$th \
        -ma=$min_area
    done
done