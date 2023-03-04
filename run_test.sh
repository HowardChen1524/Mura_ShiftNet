#!/bin/bash
# /hcds_vol/private/howard/mura_data/typecplus/img/

declare -a measure_list=(
                        #  "MSE_sliding"
                        #  "Mask_MSE_sliding"
                        #  "SSIM_sliding"
                        #  "Mask_SSIM_sliding"
                        #  "Dis_sliding"
                        #  "Mask_Dis_sliding"
                        #  "Style_VGG16_sliding"
                        #  "Mask_Style_VGG16_sliding"
                         "Content_VGG16_sliding"
                        #  "Mask_Content_VGG16_sliding"
                        )
declare th_list=(0.015)
declare min_area_list=(40)
# declare -a sup_model_list=(
#     '/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_0.pt'
#     '/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_1.pt'
#     '/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_2.pt'
# )

sup_model_path='/home/sallylab/Howard/models/SEResNeXt101_d23/model.pt'
# sup_model_version="ensemble_d23"
sup_model_version="SEResNeXt101_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
model_version="ShiftNet_SSIM_d23_8k_change_cropping"

# model_version="PEN-NET_d23_8k"

# dataset_version="mura_d23_4k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# sup_data_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_4k/" # for unsupervised model
# normal_num=5
# smura_num=5
# normal_num=16295
# smura_num=181

# dataset_version="typec_4k"
# # sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# # sup_data_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/Howard/Mura_ShiftNet/imgs/typec4k/" # for unsupervised model
# normal_num=1
# smura_num=3

# dataset_version="typec+b1"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typec+b1/img/" # for unsupervised model
# # unsup_test_normal_path="/home/levi/mura_data/d23/1920x1080/test/test_normal_8k/"
# # unsup_test_smura_path="/home/levi/mura_data/typecplus/img/"
# normal_num=0
# smura_num=31

dataset_version="typec+b1_edge"
unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/sallylab/min/typec+b1_edge/" # for unsupervised model
# unsup_test_normal_path="/home/levi/mura_data/d23/1920x1080/test/test_normal_8k/"
# unsup_test_smura_path="/home/levi/mura_data/typecplus/img/"
normal_num=0
smura_num=4

# dataset_version="typed"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
# normal_num=0
# smura_num=52

# d23 test
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

# d24 d25 blind test
# dataset_version="d24_d25_8k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d25_merge/"
# # data_csv_path="/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv"
# data_csv_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d25_data_merged_new.csv"
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/"
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/"
# normal_num=87
# smura_num=85
# conf_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k'
# # score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/ShiftNet_SSIM_d23_8k/d24_d25_8k/Mask_MSE_sliding'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/PEN-Net_d23_8k/d24_d25_8k/Mask_MSE_sliding'

# only supervised model
# for sup_model in ${sup_model_list[@]}
# do
#     python3 sup_gen_res.py \
#     --sup_model_version=$sup_model_version --data_version=$dataset_version \
#     --sup_model_path=$sup_model --sup_dataroot=$sup_data_path --data_csv_path=$data_csv_path \
#     --gpu_id=0
# done

# only unsupervised model
# for measure in ${measure_list[@]}
# do
#     python3 test_sliding.py \
#     --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
#     --loadSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --="RGB" \
#     --inpainting_mode="ShiftNet" --measure_mode=$measure --checkpoints_dir='/home/sallylab/Howard/models/' --results_dir='./exp_result/Unsupervised' --model_version=$model_version --which_epoch="200" \
#     --data_version=$dataset_version \
#     --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
#     --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
#     --gpu_ids=0 
# done

# supervised with unsupervised
# for measure in ${measure_list[@]}
# do
#     python3 sup_unsup_gen_res.py \
#     --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
#     --loadSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 \
#     --inpainting_mode="ShiftNet" --measure_mode=$measure --checkpoints_dir='./log' --model_version=$model_version --which_epoch="400" \
#     --data_version $dataset_version \
#     --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
#     --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
#     --sup_model_path=$sup_model --sup_dataroot=$sup_data_path --data_csv_path=$data_csv_path --gpu_ids=0
# done

# find line
# for measure in ${measure_list[@]}
# do
#     python3 sup_unsup_find_th_or_test.py \
#     --measure_mode=$measure --model_version=$model_version --data_version $dataset_version \
#     --conf_csv_dir=$conf_csv_dir \
#     --score_csv_dir=$score_csv_dir \
#     --checkpoints_dir='./log' --results_dir='./exp_result/Ensemble' --using_threshold
# done

# generate unsupervised model diff visualize
for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        python3 gen_patch.py \
        --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
        --data_version=$dataset_version --dataset_mode="aligned_sliding" --loadSize=64 --crop_stride=16  --mask_type="center" --input_nc=3 --output_nc=3 \
        --model_version=$model_version --which_epoch="200" --inpainting_mode="ShiftNet" --measure_mode="Content_VGG16_sliding" \
        --checkpoints_dir='/home/sallylab/Howard/models/' --results_dir='./exp_result/Unsupervised' \
        --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
        --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
        --gpu_ids=1 \
        --binary_threshold=$th --min_area=$min_area --flip_edge
    done
done

# generate gradcam
# python3 sup_gradcam.py \
# --batchSize=1 \
# --sup_model_version=$sup_model_version --checkpoints_dir='/home/sallylab/Howard/models/' \
# --data_version=$dataset_version --loadSize=64 --testing_smura_dataroot=$unsup_test_smura_path \
# --sup_gradcam_th=0.5 --gpu_ids=0

