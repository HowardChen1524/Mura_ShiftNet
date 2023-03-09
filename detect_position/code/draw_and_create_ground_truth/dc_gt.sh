#!/bin/bash
dataset_version='typec+b1'
# dataset_version='typed'
data_dir='/home/sallylab/min/'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/'

python dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir