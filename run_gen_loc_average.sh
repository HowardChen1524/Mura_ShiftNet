#!/bin/bash
# ===== basic =====
base_dir="/home/sallylab/Howard/Mura_ShiftNet/detect_position"
checkpoints_dir="../models/"
results_dir="./detect_position/"
loadSize=64
measure_mode="MAE"
gpu_ids=0
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
declare min_area_list=(1 5 10 15 20 25 30 35 40 45 50 55 60 65 70)
declare grad_th_list=(0.1)

# ===== dataset =====
dataset_version="typec+b1"
unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_8k/" # for unsupervised model
unsup_test_smura_path="/home/sallylab/min/typec+b1/img/" # for unsupervised model
normal_num=0
smura_num=31

# dataset_version="typed"
# unsup_test_normal_path="/home/sallylab/min/typed_normal/test/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed/img/" # for unsupervised model
# normal_num=0
# smura_num=26

# dataset_version="typed_shifted_7"
# unsup_test_normal_path="/home/sallylab/min/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/home/sallylab/min/typed_shifted_7/img/" # for unsupervised model
# normal_num=0
# smura_num=26

# ===== generate ground truth =====
data_dir='/home/sallylab/min/'
save_dir='/home/sallylab/Howard/Mura_ShiftNet/detect_position/'
python3 /home/sallylab/Howard/Mura_ShiftNet/detect_position/code/draw_and_create_ground_truth/dc_gt.py \
-dv=$dataset_version \
-dd=$data_dir \
-sd=$save_dir \
-rs=$isResize

# ===== unsup =====
for th in $(seq 0.010 0.010 0.100)
do
    for min_area in ${min_area_list[@]}
    do
        # generate unsupervised model diff visualize
        python3 gen_patch.py \
        --data_version=$dataset_version --loadSize=$loadSize --crop_stride=$crop_stride \
        --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
        --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
        --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure_mode \
        --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
        --resolution=$resolution --overlap_strategy=$overlap_strategy\
        --top_k=$th --min_area=$min_area \
        --gpu_ids=$gpu_ids \
        --isPadding=$isPadding

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

data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/"
save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
-dd=$data_dir \
-sd=$save_dir \
-os=$overlap_strategy

# ===== combine sup =====
for grad_th in ${grad_th_list[@]}
do
    # generate gradcam
    python3 sup_gradcam.py \
    --data_version=$dataset_version --loadSize=$loadSize --testing_smura_dataroot=$unsup_test_smura_path \
    --sup_model_version=$sup_model_version \
    --checkpoints_dir=$checkpoints_dir \
    --resolution=$resolution \
    --sup_gradcam_th=$grad_th \
    --gpu_ids=$gpu_ids

    # plot gt
    data_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${grad_th}"
    gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
    csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
    save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${grad_th}_gt"
    python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
    -cp=$csv_path \
    -dd=$data_dir \
    -gd=$gt_dir \
    -sd=$save_dir \
    -rs=$isResize # for resizing

    # cal dice and recall & precision
    gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
    save_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${grad_th}_gt"
    python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
    -dd=$data_dir \
    -gd=$gt_dir \
    -sd=$save_dir

    for th in ${th_list[@]}
    do
        for min_area in ${min_area_list[@]}
        do
            # combine gradcam
            sup_dir="${base_dir}/${dataset_version}/${resolution}/sup_gradcam/${sup_model_version}/${grad_th}"
            unsup_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/${overlap_strategy}/${th}_diff_pos_area_${min_area}/imgs"
            save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
            python3 ./detect_position/code/combine_gradcam/combine_gradcam.py \
            -upd=$unsup_dir \
            -spd=$sup_dir \
            -sd=$save_dir

            # plot gt
            data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs"
            gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
            csv_path="${base_dir}/${dataset_version}/${dataset_version}.csv"
            save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}/sup_${grad_th}_unsup_${th}_${min_area}/imgs_gt"
            python3 ./detect_position/code/plot_gt_on_result/plot_gt_on_result.py \
            -cp=$csv_path \
            -dd=$data_dir \
            -gd=$gt_dir \
            -sd=$save_dir \
            -rs=$isResize # for resizing

            # cal dice and recall & precision
            gt_dir="${base_dir}/${dataset_version}/${resolution}/actual_pos/ground_truth"
            save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}/sup_${grad_th}_unsup_${th}_${min_area}"
            python3 ./detect_position/code/calculate_metrics/calculate_metrics.py \
            -dd=$data_dir \
            -gd=$gt_dir \
            -sd=$save_dir
        done
    done
done

data_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}/combine_grad_${overlap_strategy}/"
save_dir="${base_dir}/${dataset_version}/${resolution}/${crop_stride}"
python3 ./detect_position/code/summary_exp_result/summary_exp_result.py \
-dd=$data_dir \
-sd=$save_dir \
-os=$overlap_strategy \
-ic


