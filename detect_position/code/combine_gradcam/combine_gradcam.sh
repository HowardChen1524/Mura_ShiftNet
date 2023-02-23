#!/bin/bash

dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'
crop_stride=16
declare th_list=(0.015)
declare min_area_list=(22 56)
declare gradcam_th_list=(0.1 0.2 0.3 0.4 0.5)

for grad_th in ${gradcam_th_list[@]}
do
    for th in ${th_list[@]}
    do
        for min_area in ${min_area_list[@]}
        do
            python combine_gradcam.py \
            -dv=$dataset_version \
            -dd=$data_dir \
            -cs=$crop_stride \
            -th=$th \
            -ma=$min_area \
            -gt=$grad_th
        done
    done
done