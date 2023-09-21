#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

checkpoints_dir='../models'
loadSize=64
niter=200
lr=0.0001
lr_policy='cosine'
crop_image_num=64
resolution="resized"
gpu_ids=0
nThreads=4

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_cropping_fixed_ori_res_smooth"
# model_version="ShiftNet_SSIM_typed_cropping_fixed_edge"
# model_version="ShiftNet_SSIM_typed_step_5000_cropping_fixed_ori_res"
# model_version="ShiftNet_SSIM_typed_step_5000_cropping_fixed_ori_res_smooth"
# model_version="ShiftNet_SSIM_typed_demura_cropping_fixed"

# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k_cropping_fixed_edge_ori_res_smooth" # v2: add blur

model_version="SkipGANomaly_typed_demura_cropping"
# model_version="ResunetGAN_typed_demura_cropping"

train_normal_path="/home/mura/mura_data/typed_demura/train_normal/*png"

python ./train_skipgan_resunet.py \
 --model_version=$model_version \
 --batchSize=1 \
 --loadSize=$loadSize \
 --niter=$niter \
 --lr=$lr \
 --lr_policy=$lr_policy \
 --checkpoints_dir=$checkpoints_dir \
 --resolution=$resolution \
 --dataroot=$train_normal_path \
 --crop_image_num=$crop_image_num \

# python ./train_sliding.py \
#  --model="shiftnet" \
#  --gan_type="vanilla" \
#  --which_model_netG="unet_shift_triple" \
#  --model_version=$model_version \
#  --batchSize=1 \
#  --loadSize=$loadSize \
#  --niter=$niter \
#  --lr=$lr \
#  --lr_policy=$lr_policy \
#  --checkpoints_dir=$checkpoints_dir \
#  --resolution=$resolution \
#  --dataroot=$train_normal_path \
#  --crop_image_num=$crop_image_num \
#  --nThreads=$nThreads \
#  --gpu_id=$gpu_ids \
 