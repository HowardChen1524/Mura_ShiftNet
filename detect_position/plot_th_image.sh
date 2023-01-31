#!/bin/bash

dataset_version='typec+b1'
model_version='ShiftNet_SSIM_d23_8k_change_cropping'
img_dir='/home/sallylab/Howard/find_position/'
save_dir='/home/sallylab/Howard/find_position/'

python plot_th_image.py \
-dv=$dataset_version \
-mv=$model_version \
-id=$img_dir \
-sd=$save_dir