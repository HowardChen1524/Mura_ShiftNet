#!/bin/bash
# /hcds_vol/private/howard/mura_data/typecplus/img/

checkpoints_dir="./log/Unsupervised/"
# checkpoints_dir="../models/"
results_dir="./exp_result/Unsupervised/"
loadSize=64
gpu_ids=0

declare -a measure_list=(
                         "MSE"
                        #  "SSIM"
                        #  "Feat"
                        #  "Style"
                         "Content"
                        )
crop_stride=32
# resolution="resized"
resolution="origin"
# which_epoch="200"
which_epoch="300"

# sup_model_path='/home/sallylab/Howard/models/SEResNeXt101_d23/model.pt'
# sup_model_version="ensemble_d23"
# sup_model_version="SEResNeXt101_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
model_version="ShiftNet_SSIM_d23_4k_step_5000_cropping_fixed_ori_res_smooth"

# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping_ori_res"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping_ori_res_v2"
# model_version="ShiftNet_SSIM_typed_cropping_fixed_edge"

# model_version="PEN-NET_d23_8k"

dataset_version="d23_4k"
unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_4k/" # for unsupervised model
unsup_test_smura_path="/home/sallylab/min/d23_merge/test/test_smura_4k/" # for unsupervised model
# normal_num=5
# smura_num=5
normal_num=16295
smura_num=181

# dataset_version="typed"
# unsup_test_normal_path="/home/sallylab/min/typed_normal/test/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
# normal_num=40
# smura_num=26

# dataset_version="typed_no_blue"
# unsup_test_normal_path="/home/sallylab/min/typed_normal/test/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed_no_blue/img/" # for unsupervised model
# normal_num=40
# smura_num=23

# dataset_version="d23_8k"
# # sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# # data_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/d23_merge/test/test_smura_8k/" # for unsupervised model
# normal_num=541
# smura_num=143
# conf_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d23_8k'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/ShiftNet_SSIM_d23_8k/d23_8k/Mask_MSE_sliding'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/PEN-Net_d23_8k/d23_8k/Mask_MSE_sliding'

# dataset_version="d24_d25_8k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d25_merge/"
# data_csv_path="/home/sallylab/min/d25_merge_8k/d25_data_merged_new.csv"
# unsup_test_normal_path="/home/sallylab/min/d25_merge_8k/d2425_all_normal_8k/"
# unsup_test_smura_path="/home/sallylab/min/d25_merge_8k/d2425_all_smura_8k/"
# normal_num=87
# smura_num=85
# conf_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/ShiftNet_SSIM_d23_8k/d24_d25_8k/Mask_MSE_sliding'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/PEN-Net_d23_8k/d24_d25_8k/Mask_MSE_sliding'

# only unsupervised model
for measure in ${measure_list[@]}
do
    python3 test_sliding.py \
    --batchSize=1 \
    --data_version=$dataset_version --loadSize=$loadSize --crop_stride=$crop_stride \
    --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure \
    --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
    --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
    --resolution=$resolution \
    --gpu_ids=$gpu_ids

    python3 test_sliding.py \
    --batchSize=1 \
    --data_version=$dataset_version --loadSize=$loadSize --crop_stride=$crop_stride \
    --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure \
    --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
    --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
    --resolution=$resolution \
    --gpu_ids=$gpu_ids \
    --mask_part

    python3 test_sliding.py \
    --batchSize=1 \
    --data_version=$dataset_version --loadSize=$loadSize --crop_stride=$crop_stride \
    --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure \
    --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
    --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
    --resolution=$resolution \
    --gpu_ids=$gpu_ids \
    --pos_normalize 
done
