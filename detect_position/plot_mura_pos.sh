#!/bin/bash
dataset_version='typec+b1'
data_dir='/home/sallylab/min/'
save_dir='/home/sallylab/Howard/detect_position/'

python plot_mura_pos.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir