#!/bin/bash

dataset_version='typec+b1'
data_dir='/home/sallylab/Howard/detect_position/'
crop_stride='32'
th_list='0.0150,0.0200,0.0250'

python compare_img_diff_threshold.py \
-dv=$dataset_version \
-dd=$data_dir \
-cs=$crop_stride \
-ths=$th_list
