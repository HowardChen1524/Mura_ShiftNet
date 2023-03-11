#!/bin/bash

# sup_model_version="ensemble_d23"
sup_model_version="SEResNeXt101_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
model_version="ShiftNet_SSIM_d23_8k_change_cropping"

base_dir="/home/sallylab/Howard/Mura_ShiftNet/detect_position"

declare th_list=(0.0150)
declare min_area_list=(40)
declare grad_th_list=(0.1 0.2 0.3 0.4 0.5)
crop_stride=16

# ===== dataset =====
dataset_version="typec+b1"
unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/sallylab/min/typec+b1/img/" # for unsupervised model
normal_num=0
smura_num=31

# dataset_version="typec+b1_edge"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typec+b1_edge/" # for unsupervised model
# # unsup_test_normal_path="/home/levi/mura_data/d23/1920x1080/test/test_normal_8k/"
# # unsup_test_smura_path="/home/levi/mura_data/typecplus/img/"
# normal_num=0
# smura_num=4

# dataset_version="typed"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
# normal_num=0
# smura_num=52

# ===== generate ground truth =====
# data_dir='/home/sallylab/min/'
# save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/'
# python3 /home/sallylab/Howard/Mura_ShiftNet/detect_position/code/draw_and_create_ground_truth/dc_gt.py \
# -dv=$dataset_version \
# -dd=$data_dir \
# -sd=$save_dir

# ===== unsup =====
# for th in ${th_list[@]}
# do
#     for min_area in ${min_area_list[@]}
#     do
#         # generate unsupervised model diff visualize
#         python3 gen_patch.py \
#         --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
#         --data_version=$dataset_version --dataset_mode="aligned_sliding" --loadSize=64 --crop_stride=$crop_stride  --mask_type="center" --input_nc=3 --output_nc=3 \
#         --model_version=$model_version --which_epoch="200" \
#         --checkpoints_dir='/home/sallylab/Howard/models/' --results_dir='./detect_position/' \
#         --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
#         --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
#         --binary_threshold=$th --min_area=$min_area --isPadding \
#         --gpu_ids=1

#         # plot gt
#         data_dir="${base_dir}/${dataset_version}/${crop_stride}/union/${th}_diff_pos_area_${min_area}/imgs"
#         csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#         save_dir="${base_dir}/${dataset_version}/${crop_stride}/union/${th}_diff_pos_area_${min_area}/imgs_gt"
#         python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#         -cp=$csv_path \
#         -dd=$data_dir \
#         -sd=$save_dir

#         # cal dice and recall & precision
#         gt_dir="${base_dir}/${dataset_version}/actual_pos/ground_truth"
#         save_dir="${base_dir}/${dataset_version}/${crop_stride}/union/${th}_diff_pos_area_${min_area}"
#         python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir
#     done
# done

data_dir="${base_dir}/${dataset_version}/${crop_stride}/union/"
python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
-dd=$data_dir

# ===== combine sup =====
# for grad_th in ${grad_th_list[@]}
# do
#     # generate gradcam
#     python3 sup_gradcam.py \
#     --batchSize=1 \
#     --sup_model_version=$sup_model_version --checkpoints_dir='/home/sallylab/Howard/models/' \
#     --data_version=$dataset_version --loadSize=64 --testing_smura_dataroot=$unsup_test_smura_path \
#     --sup_gradcam_th=$grad_th \
#     --gpu_ids=1

#     for th in ${th_list[@]}
#     do
#         for min_area in ${min_area_list[@]}
#         do
#             # combine gradcam
#             sup_dir="${base_dir}/${dataset_version}/sup_gradcam/${sup_model_version}/${grad_th}"
#             unsup_dir="${base_dir}/${dataset_version}/${crop_stride}/union/${th}_diff_pos_area_${min_area}/imgs"
#             save_dir="${base_dir}/${dataset_version}/${crop_stride}/combine_grad/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
#             python3 ./detect_position/code/combine_gradcam/combine_gradcam.py \
#             -upd=$unsup_dir \
#             -spd=$sup_dir \
#             -sd=$save_dir

#             # plot gt
#             data_dir="${base_dir}/${dataset_version}/${crop_stride}/combine_grad/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
#             csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#             save_dir="${base_dir}/${dataset_version}/${crop_stride}/combine_grad/sup_${grad_th}_unsup_${th}_${min_area}/imgs_gt"
#             python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#             -cp=$csv_path \
#             -dd=$data_dir \
#             -sd=$save_dir

#             # cal dice and recall & precision
#             gt_dir="${base_dir}/${dataset_version}/actual_pos/ground_truth"
#             save_dir="${base_dir}/${dataset_version}/${crop_stride}/combine_grad/sup_${grad_th}_unsup_${th}_${min_area}"
#             python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#             -dd=$data_dir \
#             -gd=$gt_dir \
#             -sd=$save_dir
#         done
#     done
# done

data_dir="${base_dir}/${dataset_version}/${crop_stride}/combine_grad/"
python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
-dd=$data_dir \
-ic


