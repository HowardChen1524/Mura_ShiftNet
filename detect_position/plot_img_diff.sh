#!/bin/bash

dataset_version='typec+b1'
model_version='ShiftNet_SSIM_d23_8k_change_cropping'
img_dir='/home/sallylab/Howard/Mura_ShiftNet/exp_result/Unsupervised/ShiftNet_SSIM_d23_8k_change_cropping/typec+b1/Content_VGG16_sliding/check_inpaint/'
save_dir='/home/sallylab/Howard/detect_position/'

python plot_img_diff.py \
-dv=$dataset_version \
-mv=$model_version \
-id=$img_dir \
-sd=$save_dir