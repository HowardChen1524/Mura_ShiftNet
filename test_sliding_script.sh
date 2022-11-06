#!/bin/bash
# /hcds_vol/private/howard/mura_data/typecplus/img/
# declare -a measure_list=(
#                          "MAE_sliding" "Mask_MAE_sliding" "MSE_sliding" "Mask_MSE_sliding"
#                          "Discounted_L1_sliding" "SSIM_sliding" "Mask_SSIM_sliding" "Dis_sliding"
#                          "Mask_Dis_sliding" "Style_VGG16_sliding" "Mask_Style_VGG16_sliding" "Content_VGG16_sliding"
#                          "Mask_Content_VGG16_sliding" 
#                         )

declare -a test_type=( 
                        # sup_unsup_gen_res.py
                        sup_unsup_find_th_or_test.py
                     )

declare -a measure_list=(
                         "Mask_MSE_sliding"
                        )

model_version="mura_d23_8k"

# d23 test
dataset_name="mura_d23_8k"
sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/" # for unsupervised model
normal_num=541
smura_num=143

# d24 d25 blind test
# dataset_name="mura_d24_d25_8k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d25_merge/"
# csv_path="/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv"
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/"
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/"
# normal_num=88
# smura_num=85

# for measure in ${measure_list[@]}
# do
#     python3 sup_unsup_gen_res.py \
#     --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
#     --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" \
#     --inpainting_mode="ShiftNet" --measure_mode=$measure --which_epoch="200" \
#     --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
#     --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
#     --minmax
# done

for measure in ${measure_list[@]}
do
    python3 sup_unsup_find_th_or_test.py \
    --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
    --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" \
    --inpainting_mode="ShiftNet" --measure_mode=$measure --which_epoch="200" \
    --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
    --minmax --using_threshold
done
