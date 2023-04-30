#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results

checkpoints_dir='./log/Unsupervised'
loadSize=64
niter=400
lr=0.0001
lr_policy='cosine'
crop_image_num=64
resolution="origin"
gpu_ids=0
nThreads=4

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
model_version="ShiftNet_SSIM_d23_4k_step_5000_cropping_fixed_ori_res_smooth"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k_cropping_fixed_edge_ori_res_smooth" # v2: add blur
# model_version="ShiftNet_SSIM_typed_cropping_fixed_edge"

# train_normal_path="/home/sallylab/min/d23_merge/train/d23_normal_8k/"
# train_normal_path="/home/sallylab/min/typed_normal/train/"
train_normal_path="/home/sallylab/min/d23_merge/train/normal_4k/"

python ./train_sliding.py \
 --model_version=$model_version \
 --batchSize=1 --loadSize=$loadSize \
 --niter=$niter --lr=$lr --lr_policy=$lr_policy \
 --checkpoints_dir=$checkpoints_dir \
 --resolution=$resolution --dataroot=$train_normal_path --crop_image_num=$crop_image_num \
 --nThreads=$nThreads --gpu_id=$gpu_ids
 