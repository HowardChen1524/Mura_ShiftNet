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

sup_model_path='/home/sallylab/Howard/models/SEResNeXt101_d23/model.pt'
# sup_model_version="ensemble_d23"
sup_model_version="SEResNeXt101_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
model_version="ShiftNet_SSIM_d23_8k_change_cropping"

crop_stride=16

# dataset_version="typec+b1"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typec+b1/img/" # for unsupervised model
# normal_num=0
# smura_num=31

# dataset_version="typec+b1_edge"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typec+b1_edge/" # for unsupervised model
# # unsup_test_normal_path="/home/levi/mura_data/d23/1920x1080/test/test_normal_8k/"
# # unsup_test_smura_path="/home/levi/mura_data/typecplus/img/"
# normal_num=0
# smura_num=4

dataset_version="typed"
unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
normal_num=0
smura_num=52

# gnerate ground truth
data_dir='/home/sallylab/min/'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/'

python /home/sallylab/Howard/Mura_ShiftNet/detect_position/code/draw_and_create_ground_truth/dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir

# generate unsupervised model diff visualize
for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        python3 gen_patch.py \
        --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
        --data_version=$dataset_version --dataset_mode="aligned_sliding" --loadSize=64 --crop_stride=$crop_stride  --mask_type="center" --input_nc=3 --output_nc=3 \
        --model_version=$model_version --which_epoch="200" --inpainting_mode="ShiftNet" --measure_mode="Content_VGG16_sliding" \
        --checkpoints_dir='/home/sallylab/Howard/models/' --results_dir='./exp_result/Unsupervised' \
        --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
        --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
        --gpu_ids=1 \
        --binary_threshold=$th --min_area=$min_area --max_area=$max_area --flip_edge
    done
done

# generate gradcam
# python3 sup_gradcam.py \
# --batchSize=1 \
# --sup_model_version=$sup_model_version --checkpoints_dir='/home/sallylab/Howard/models/' \
# --data_version=$dataset_version --loadSize=64 --testing_smura_dataroot=$unsup_test_smura_path \
# --sup_gradcam_th=0.5 --gpu_ids=0

# calculate dice
res_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'

for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        for max_area in ${max_area_list[@]}
        do
            python /home/sallylab/Howard/Mura_ShiftNet/detect_position/code/calculate_dice/calculate_dice.py \
            -dv=$dataset_version \
            -dd=$res_dir \
            -cs=$crop_stride \
            -th=$th \
            -mi=$min_area \
            -mx=$max_area
        done
    done
done

declare th_list=(0.0150)
# calculate recall precision
gt_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/actual_pos/ground_truth'
for th in ${th_list[@]}
do
    for min_area in ${min_area_list[@]}
    do
        for max_area in ${max_area_list[@]}
        do
            res_dir="/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/16/union/${th}_diff_pos_area_${min_area}_${max_area}"
            
            python /home/sallylab/Howard/Mura_ShiftNet/detect_position/code/calculate_pixel_based_recall_precision/calculate_recall_precision.py \
            -dv=$dataset_version \
            -dd=$res_dir \
            -gd=$gt_dir
        done
    done
done

data_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/16/union/0.0150_diff_pos_area_40_80'
csv_path='/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/typec+b1.csv'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position'

python /home/sallylab/Howard/Mura_ShiftNet/detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
-dv=$dataset_version \
-cp=$csv_path \
-dd=$data_dir \
-sd=$save_dir
