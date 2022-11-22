#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results
# which_model_netD selects model to use for netD, [basic|densenet]
# which_model_netG selects model to use for netG [unet_256| unet_shift_triple| \
#                                                 res_unet_shift_triple|patch_soft_unet_shift_triple| \
#                                                 res_patch_soft_unet_shift_triple| face_unet_shift_triple]
# model chooses which model to use. [shiftnet|res_shiftnet|patch_soft_shiftnet|res_patch_soft_shiftnet|test]
# shift_sz shift_sz>1 only for \'soft_shift_patch\'.'
# mask_thred number to decide whether a patch is masked

model_version="ShiftNet_SSIM_d23_4k"
train_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/train/normal_4k/"

python ./train_sliding.py \
--batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--niter=200 --lr=0.0005 --lr_policy=cosine --random_choose_num=5000 --crop_image_num=64 --resolution='resized' \
--checkpoints_dir='./log'  --model_version=$model_version
--dataroot=$train_normal_path