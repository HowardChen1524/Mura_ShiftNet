#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping"
model_version="ShiftNet_SSIM_d23_8k_change_cropping_ori_res_v2" # v2: add blur

train_normal_path="/home/sallylab/min/d23_merge/train/d23_normal_8k/"

python ./train_sliding.py \
--batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 \
--niter=300 --lr=0.0001 --lr_policy=cosine --fix_step=5000 --crop_image_num=64 \
--checkpoints_dir='./log/Unsupervised'  --model_version=$model_version \
--resolution='origin' \
--gpu_id=1 --nThreads=4 \
--dataroot=$train_normal_path