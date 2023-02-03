#!/bin/bash

dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/detect_position/'
crop_stride='32'
declare th_list=(
                '0.0100'
                '0.0150'
                '0.0200'
                '0.0225'
                '0.0250'
                '0.0275'
                '0.0300'
                )

for th in ${th_list[@]}
do
    python plot_actual_diff_img.py \
    -dv=$dataset_version \
    -dd=$data_dir \
    -cs=$crop_stride \
    -th=$th
done