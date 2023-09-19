#!/bin/bash
# exp
# dataset
# model
# cls loc

# ===== basic =====
checkpoints_dir="/home/mura/models"
gpu_ids="0"

# ===== prepro =====
loadSize=64
crop_stride=16
resolution="resized"
isResize=1
isPadding=1
# crop_stride=32
# resolution="origin"
# isResize=0
# isPadding=0

# ===== model =====
# sup_model_version="ensemble_d23"
# sup_model_version="SEResNeXt101_d23"
sup_model_version="SEResNeXt101_d23_8k"
# model_version="ShiftNet_SSIM_d23_8k_cropping_fixed"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="SkipGANomaly_d23_8k_cropping"
# model_version="ResunetGAN_d23_8k_cropping"
model_version="CSA_d23_8k_cropping_fixed"

# model="shiftnet"
model="csa"
# gan_type="vanilla"
gan_type="lsgan"

# which_model_netG="unet_shift_triple"
which_model_netG="unet_csa"

which_epoch="177"
# which_epoch="200"

# ===== find Mura =====
measure_mode="MSE"
overlap_strategy="average"
# declare min_area_range=(10 15 20 25 30 35 40 45 50)
# declare top_k_range=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.10)
# declare min_area_range=(10 15)
# declare grad_range=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

declare min_area_range=(10)
declare top_k_range=(0.01)
# declare grad_range=(0.4)

# ===== dataset =====
dataset_mode="aligned_sliding"
dataset_version="typec+b1"
test_smura_path="/home/mura/mura_data/typec+b1/img"
smura_num=31
seg_gt_dir=/home/mura/mura_data/typec+b1/seg_gt
bb_gt_dir=/home/mura/mura_data/typec+b1/bb_gt
csv_dir=/home/mura/mura_data/typec+b1/Mura_type_c_plus.csv

# ===== Shift-Net =====
for top_k in "${top_k_range[@]}";
do
    for min_area in "${min_area_range[@]}";
    do
        results_dir="../thesis_exp/${dataset_version}/${model_version}/loc/${top_k}_percent_min_area_${min_area}"
        mkdir -p $results_dir
        
        python3 gen_patch_vis.py \
        --data_version=$dataset_version --dataset_mode=$dataset_mode --loadSize=$loadSize --crop_stride=$crop_stride \
        --smura_how_many=$smura_num --testing_smura_dataroot=$test_smura_path \
        --model=$model --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure_mode \
        --gan_type=$gan_type --which_model_netG=$which_model_netG \
        --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
        --resolution=$resolution --overlap_strategy=$overlap_strategy\
        --top_k=$top_k --min_area=$min_area --isPadding=$isPadding \
        --gpu_ids=$gpu_ids

        # plot gt
        # data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs"
        # gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
        # csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
        # save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs_gt"
        # python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
        # -cp=$csv_path \
        # -dd=$data_dir \
        # -gd=$gt_dir \
        # -sd=$save_dir \
        # -rs=$isResize # for resizing

        # cal dice and recall & precision
        # data_dir="${results_dir}/img"
        # python3 ./vis_code/calculate_metrics/calculate_metrics.py \
        # -dd=$data_dir \
        # -sgd=$seg_gt_dir \
        # -sd=$results_dir \
        # -ir=$isResize \
        # -cd=$csv_dir \
        # -bgd=$bb_gt_dir
    done
done

# data_dir="../thesis_exp/${dataset_version}/${model_version}/loc"
# python3 ./vis_code/summary_exp_result/summary_exp_result.py \
# -dd=$data_dir \
# -ct="unsup"

# ===== combine sup =====

# for grad_th in "${grad_range[@]}";
# do
#     results_dir="../thesis_exp/${dataset_version}/${sup_model_version}/loc/grad_${grad_th}"
#     mkdir -p $results_dir

#     # python3 gen_gradcam.py \
#     # --data_version=$dataset_version \
#     # --testing_smura_dataroot=$test_smura_path \
#     # --sup_model_version=$sup_model_version \
#     # --checkpoints_dir=$checkpoints_dir \
#     # --results_dir=$results_dir \
#     # --resolution=$resolution \
#     # --sup_gradcam_th=$grad_th \
#     # --gpu_ids=$gpu_ids

#     # # plot gt
#     # data_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs"
#     # gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#     # csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#     # save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs_gt"
#     # python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#     # -cp=$csv_path \
#     # -dd=$data_dir \
#     # -gd=$gt_dir \
#     # -sd=$save_dir \
#     # -rs=$isResize # for resizing

#     # cal dice and recall & precision
#     # data_dir="${results_dir}/img"
#     # python3 ./vis_code/calculate_metrics/calculate_metrics.py \
#     # -dd=$data_dir \
#     # -sgd=$seg_gt_dir \
#     # -sd=$results_dir \
#     # -ir=$isResize \
#     # -cd=$csv_dir \
#     # -bgd=$bb_gt_dir

#     for top_k in "${top_k_range[@]}";
#     do
#         for min_area in "${min_area_range[@]}";
#         do
#             # combine gradcam
#             sup_dir="../thesis_exp/${dataset_version}/${sup_model_version}/loc/grad_${grad_th}/img"
#             unsup_dir="../thesis_exp/${dataset_version}/${model_version}/loc/${top_k}_percent_min_area_${min_area}/img"
#             results_dir="../thesis_exp/${dataset_version}/${sup_model_version}_${model_version}/loc/grad_${grad_th}_${top_k}_percent_min_area_${min_area}"
#             mkdir -p $results_dir
#             python3 ./vis_code/combine_gradcam/combine_gradcam.py \
#             -upd=$unsup_dir \
#             -spd=$sup_dir \
#             -sd=$results_dir

#             # plot gt
#             # data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
#             # gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#             # csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#             # save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs_gt"
#             # python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#             # -cp=$csv_path \
#             # -dd=$data_dir \
#             # -gd=$gt_dir \
#             # -sd=$save_dir \
#             # -rs=$isResize # for resizing

#             # cal dice and recall & precision
#             data_dir="${results_dir}/img"
#             python3 ./vis_code/calculate_metrics/calculate_metrics.py \
#             -dd=$data_dir \
#             -sgd=$seg_gt_dir \
#             -sd=$results_dir \
#             -ir=$isResize \
#             -cd=$csv_dir \
#             -bgd=$bb_gt_dir

#         done
#     done        
# done

# data_dir="../thesis_exp/${dataset_version}/${sup_model_version}/loc"
# python3 ./vis_code/summary_exp_result/summary_exp_result.py \
# -dd=$data_dir \
# -ct="sup"

# data_dir="../thesis_exp/${dataset_version}/${sup_model_version}_${model_version}/loc/"
# python3 ./vis_code/summary_exp_result/summary_exp_result.py \
# -dd=$data_dir \
# -ct="combine"



