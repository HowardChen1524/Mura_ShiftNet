#!/bin/bash
# use_spectral_norm_D whether to add spectral norm to D, it helps improve results
# which_model_netD selects model to use for netD, [basic|densenet]
# which_model_netG selects model to use for netG [unet_256| unet_shift_triple| \
#                                                 res_unet_shift_triple|patch_soft_unet_shift_triple| \
#                                                 res_patch_soft_unet_shift_triple| face_unet_shift_triple]
# model chooses which model to use. [shiftnet|res_shiftnet|patch_soft_shiftnet|res_patch_soft_shiftnet|test]
# shift_sz shift_sz>1 only for \'soft_shift_patch\'.'
# mask_thred number to decide whether a patch is masked

# model_version="ShiftNet_SSIM_d23_4k"
model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_d23_8k"

train_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/train/normal_4k/"
# train_normal_path="/hcds_vol/private/howard/mura_data/d23/train/normal_8k/"

python ./train_sliding.py \
--batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' --resolution='resized' \
--niter=200 --lr=0.0001 --lr_policy=cosine --fix_step=5000 --crop_image_num=64 \
--checkpoints_dir='./log'  --model_version=$model_version \
--gpu_id=0 --nThreads=4 \
--dataroot=$train_normal_path

dataset_version="d23_4k"
unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_4k/" # for unsupervised model
unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_4k/" # for unsupervised model
# normal_num=5
# smura_num=5
normal_num=16295
smura_num=181

declare -a measure_list=(
                         "MSE_sliding"
                         "Mask_MSE_sliding"
                        #  "SSIM_sliding"
                        #  "Mask_SSIM_sliding"
                        #  "Dis_sliding"
                        #  "Mask_Dis_sliding"
                        #  "Style_VGG16_sliding"
                        #  "Mask_Style_VGG16_sliding"
                         "Content_VGG16_sliding"
                         "Mask_Content_VGG16_sliding"
                        )

for measure in ${measure_list[@]}
do
    python3 test_sliding.py \
    --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
    --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" \
    --inpainting_mode="ShiftNet" --measure_mode=$measure --checkpoints_dir='./log' --model_version=$model_version --which_epoch="200" \
    --data_version=$dataset_version \
    --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
    --gpu_ids=0
done

for measure in ${measure_list[@]}
do
    python3 test_sliding.py \
    --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
    --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" \
    --inpainting_mode="ShiftNet" --measure_mode=$measure --checkpoints_dir='./log' --model_version=$model_version --which_epoch="200" \
    --data_version=$dataset_version \
    --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
    --gpu_ids=0 --pos_normalize
done