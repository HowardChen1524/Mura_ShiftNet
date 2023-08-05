#!/bin/bash
# ===== basic =====
base_dir="/home/mura/Mura_ShiftNet/detect_position"
checkpoints_dir="../models/"
results_dir="./detect_position/"
loadSize=64
measure_mode="MAE"
gpu_ids="0"

# ===== model =====
# sup_model_version="ensemble_d23"
sup_model_version="SEResNeXt101_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
model_version="ShiftNet_SSIM_typed_step_5000_cropping_fixed_ori_res_smooth"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping_ori_res"
# model_version="ShiftNet_SSIM_d23_8k_cropping_fixed_edge_ori_res_smooth"
# model_version="ShiftNet_SSIM_typed_cropping_fixed_edge"

# which_epoch="200"
which_epoch="300"
# which_epoch="500"

# ===== find Mura =====
# crop_stride=16
crop_stride=32
# resolution="resized"
resolution="origin"
overlap_strategy="average"
# isPadding=1
isPadding=0
# isResize=1
isResize=0
# sup_th_strategy='dynamic'
# sup_th_strategy='fixed'
declare min_area_list=(15)
# declare grad_th_list=(0.1)

# ===== dataset =====
dataset_mode="aligned_sliding"
# dataset_version="typec+b1"
# unsup_test_normal_path="/home/mura/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/home/mura/mura_data/typec+b1/img/" # for unsupervised model
# normal_num=0
# smura_num=31

# dataset_version="typed"
# unsup_test_normal_path="/home/sallylab/min/typed_normal/test/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
# normal_num=0
# smura_num=26

dataset_version="typed_demura"
unsup_test_normal_path="/home/mura/mura_data/typed_demura/test_normal" # for unsupervised model
unsup_test_smura_path="/home/mura/mura_data/typed_demura/test_smura" # for unsupervised model
normal_num=0
smura_num=26

# dataset_version="typed_shifted_7"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed_shifted_7/img/" # for unsupervised model
# normal_num=0
# smura_num=26

# ===== generate ground truth =====
data_dir='/home/mura/mura_data'
save_dir='/home/mura/Mura_ShiftNet/detect_position/'
python3 /home/mura/Mura_ShiftNet/detect_position/code/draw_and_create_ground_truth/dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir \
-rs=$isResize

# ===== unsup =====
for th in $(seq 0.010 0.010 0.010)
do
    for min_area in ${min_area_list[@]}
    do
        # generate unsupervised model diff visualize
        python3 gen_patch.py \
        --data_version=$dataset_version --dataset_mode=$dataset_mode --loadSize=$loadSize --crop_stride=$crop_stride \
        --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
        --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
        --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure_mode \
        --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
        --resolution=$resolution --overlap_strategy=$overlap_strategy\
        --top_k=$th --min_area=$min_area --isPadding=$isPadding \
        --gpu_ids=$gpu_ids

        # plot gt
        data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs"
        gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
        csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
        save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs_gt"
        python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
        -cp=$csv_path \
        -dd=$data_dir \
        -gd=$gt_dir \
        -sd=$save_dir \
        -rs=$isResize # for resizing

        # cal dice and recall & precision
        gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
        save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}"
        python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
        -dd=$data_dir \
        -gd=$gt_dir \
        -sd=$save_dir
    done
done

# data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/"
# save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
# python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
# -dd=$data_dir \
# -sd=$save_dir \
# -os=$overlap_strategy

# ===== combine sup =====
# if [ $sup_th_strategy == 'dynamic' ];
# then
#     for topk in $(seq 0.01 0.01 0.10)
#     do
#         # generate gradcam
#         python3 sup_gradcam.py \
#         --data_version=$dataset_version --loadSize=$loadSize --testing_smura_dataroot=$unsup_test_smura_path \
#         --sup_model_version=$sup_model_version \
#         --checkpoints_dir=$checkpoints_dir \
#         --resolution=$resolution \
#         --sup_th_strategy=$sup_th_strategy \
#         --top_k=$topk \
#         --gpu_ids=$gpu_ids

#         # plot gt
#         data_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${topk}/imgs"
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${topk}/imgs_gt"
#         python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#         -cp=$csv_path \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir \
#         -rs=$isResize # for resizing

#         # cal dice and recall & precision
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${topk}"
#         python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir

#         for th in ${th_list[@]}
#         do
#             for min_area in ${min_area_list[@]}
#             do
#                 # combine gradcam
#                 sup_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${topk}/imgs"
#                 unsup_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${th}_${min_area}/imgs"
#                 python3 ./detect_position/code/combine_gradcam/combine_gradcam.py \
#                 -upd=$unsup_dir \
#                 -spd=$sup_dir \
#                 -sd=$save_dir

#                 # plot gt
#                 data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${th}_${min_area}/imgs"
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${th}_${min_area}/imgs_gt"
#                 python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#                 -cp=$csv_path \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir \
#                 -rs=$isResize # for resizing

#                 # cal dice and recall & precision
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${topk}_unsup_${th}_${min_area}"
#                 python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir
#             done
#         done
#     done

#     data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/"
#     save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
#     python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
#     -dd=$data_dir \
#     -sd=$save_dir \
#     -os=$overlap_strategy \
#     -ic
    
# elif [ $sup_th_strategy == 'fixed' ];
# then
#     for grad_th in ${grad_th_list[@]}
#     do
#         # generate gradcam
#         python3 sup_gradcam.py \
#         --data_version=$dataset_version --loadSize=$loadSize --testing_smura_dataroot=$unsup_test_smura_path \
#         --sup_model_version=$sup_model_version \
#         --checkpoints_dir=$checkpoints_dir \
#         --resolution=$resolution \
#         --sup_th_strategy=$sup_th_strategy \
#         --sup_gradcam_th=$grad_th \
#         --gpu_ids=$gpu_ids

#         # plot gt
#         data_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs"
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs_gt"
#         python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#         -cp=$csv_path \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir \
#         -rs=$isResize # for resizing

#         # cal dice and recall & precision
#         gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#         save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${sup_th_strategy}/${grad_th}"
#         python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#         -dd=$data_dir \
#         -gd=$gt_dir \
#         -sd=$save_dir

#         for th in ${th_list[@]}
#         do
#             for min_area in ${min_area_list[@]}
#             do
#                 # combine gradcam
#                 sup_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam//${sup_model_version}/${sup_th_strategy}/${grad_th}/imgs"
#                 unsup_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
#                 python3 ./detect_position/code/combine_gradcam/combine_gradcam.py \
#                 -upd=$unsup_dir \
#                 -spd=$sup_dir \
#                 -sd=$save_dir

#                 # plot gt
#                 data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs_gt"
#                 python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
#                 -cp=$csv_path \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir \
#                 -rs=$isResize # for resizing

#                 # cal dice and recall & precision
#                 gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
#                 save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/sup_${grad_th}_unsup_${th}_${min_area}"
#                 python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
#                 -dd=$data_dir \
#                 -gd=$gt_dir \
#                 -sd=$save_dir
#             done
#         done        
#     done

#     data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}_${sup_th_strategy}/"
#     save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
#     python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
#     -dd=$data_dir \
#     -sd=$save_dir \
#     -os=$overlap_strategy \
#     -ic
# fi



