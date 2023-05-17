#!/bin/bash
# ===== basic =====
base_dir="/home/mura/Mura_ShiftNet/detect_position"
checkpoints_dir="../models/"
results_dir="./detect_position/"
loadSize=64
measure_mode="MAE"
gpu_ids="-1" # cpu

# ===== model =====
# sup_model_version="ensemble_d23"
sup_model_version="SEResNeXt101_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_4k_step_5000_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
model_version="ShiftNet_SSIM_d23_8k_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k_change_cropping_ori_res"
# model_version="ShiftNet_SSIM_d23_8k_cropping_fixed_edge_ori_res_smooth"
# model_version="ShiftNet_SSIM_typed_cropping_fixed_edge"

which_epoch="200"
# which_epoch="300"
# which_epoch="500"

# ===== find Mura =====
crop_stride=16
# crop_stride=32
resolution="resized"
# resolution="origin"
overlap_strategy="average"
isPadding=1
# isPadding=0
isResize=1
# isResize=0
declare min_area_list=(10 15 20 25 30)
# declare min_area_list=(1)

# ===== dataset =====
dataset_mode="sup_unsup_dataset"
dataset_version="typec+b1"
unsup_test_normal_path="/home/mura/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/mura/mura_data/typec+b1/img/" # for unsupervised model
normal_num=0
smura_num=31

# dataset_version="typed"
# unsup_test_normal_path="/home/sallylab/min/typed_normal/test/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
# normal_num=0
# smura_num=26

# ===== generate ground truth =====
# data_dir='/home/mura/mura_data'
# save_dir='/home/mura/Mura_ShiftNet/detect_position/'
# python3 /home/mura/Mura_ShiftNet/detect_position/code/draw_and_create_ground_truth/dc_gt.py \
# -dv=$dataset_version \
# -dd=$data_dir \
# -sd=$save_dir \
# -rs=$isResize

# ===== unsup =====
for th in $(seq 0.010 0.010 0.100)
do
    for min_area in ${min_area_list[@]}
    do
        for beta in $(seq 0.0 1.0 1.0)
        do
            for alpha in $(seq 0.0 500.0 10000.0)
            do # 10e-6 10e-2
                if [ $alpha != 0.0 ] && [ $beta != 0.0 ]; 
                then
                    # generate unsupervised model diff visualize
                    python3 combine_gen_patch.py \
                    --data_version=$dataset_version --dataset_mode=$dataset_mode --loadSize=$loadSize --crop_stride=$crop_stride \
                    --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
                    --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure_mode \
                    --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
                    --resolution=$resolution --overlap_strategy=$overlap_strategy\
                    --isPadding=$isPadding \
                    --sup_model_version=$sup_model_version \
                    --top_k=$th --min_area=$min_area \
                    --combine_alpha=$alpha --combine_beta=$beta \
                    --gpu_ids=$gpu_ids

                    # plot gt
                    data_dir="${base_dir}/${dataset_version}/${resolution}/combine_sup_unsup/combine/${th}_diff_pos_area_${min_area}_${alpha}_${beta}/imgs"
                    gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
                    csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
                    save_dir="${base_dir}/${dataset_version}/${resolution}/combine_sup_unsup/combine/${th}_diff_pos_area_${min_area}_${alpha}_${beta}/imgs_gt"
                    python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
                    -cp=$csv_path \
                    -dd=$data_dir \
                    -gd=$gt_dir \
                    -sd=$save_dir \
                    -rs=$isResize # for resizing

                    # cal dice and recall & precision
                    gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
                    save_dir="${base_dir}/${dataset_version}/${resolution}/combine_sup_unsup/combine/${th}_diff_pos_area_${min_area}_${alpha}_${beta}_sup_${grad_th}"
                    python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
                    -dd=$data_dir \
                    -gd=$gt_dir \
                    -sd=$save_dir
                fi
            done
        done
    done
done

data_dir="${base_dir}/${dataset_version}/${resolution}/combine_sup_unsup/combine/"
save_dir="${base_dir}/${dataset_version}/${resolution}/combine_sup_unsup/"
python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
-dd=$data_dir \
-sd=$save_dir \
-os=$overlap_strategy
