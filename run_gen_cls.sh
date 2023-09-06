#!/bin/bash

# ===== basic =====
checkpoints_dir="/home/mura/models"
gpu_ids="0"

# ===== prepro =====
loadSize=64
# crop_stride=16
resolution="resized"
isResize=1
# isPadding=1
crop_stride=32
# resolution="origin"
# isResize=0
isPadding=0

# ===== model =====
# sup_model_version="ensemble_d23"
# sup_model_version="SEResNeXt101_d23"
sup_model_version="SEResNeXt101_d23_8k"
model_version="ShiftNet_SSIM_d23_8k_change_cropping"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="SkipGANomaly_d23_8k_cropping"
# model_version="ResunetGAN_d23_8k_cropping"

which_epoch="200"

# ===== classify Mura =====
measure="MSE"
mask_part=1

# ===== Dataset =====
dataset_mode="aligned_sliding"
dataset_version="d23_8k"
test_normal_path="/home/mura/mura_data/d23_merge/test/test_normal_8k" 
test_smura_path="/home/mura/mura_data/d23_merge/test/test_smura_8k"
normal_num=541
smura_num=143
sup_data_path="/home/mura/mura_data/d23_merge_8k"
data_csv_path="/home/mura/mura_data/d23_merge_8k/data_merged.csv"

# ===== Unsupervised 生成 anomaly score =====
# if [ "$mask_part" -eq 1 ]; then
#     results_dir="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}_mask"
# else
#     results_dir="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}"
# fi
# mkdir -p $results_dir
# python3 gen_unsup_score.py \
# --batchSize=1 \
# --data_version=$dataset_version --dataset_mode=$dataset_mode --loadSize=$loadSize --crop_stride=$crop_stride \
# --model_version=$model_version --which_epoch=$which_epoch --measure_mode=$measure \
# --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir \
# --normal_how_many=$normal_num --testing_normal_dataroot=$test_normal_path \
# --smura_how_many=$smura_num --testing_smura_dataroot=$test_smura_path \
# --resolution=$resolution --isPadding=$isPadding \
# --gpu_ids=$gpu_ids \
# --mask_part=$mask_part

## Skip-GANomaly, Res-unetGAN


# ===== Supervised 生成 confidence score =====
# results_dir="../thesis_exp/${dataset_version}/${sup_model_version}/cls"
# mkdir -p ${results_dir}
# python3 gen_sup_score.py \
# --sup_model_version=$sup_model_version --data_version=$dataset_version \
# --checkpoints_dir=$checkpoints_dir --results_dir=$results_dir --sup_model_path=$sup_model \
# --sup_dataroot=$sup_data_path --data_csv_path=$data_csv_path \
# --gpu_id=0

# ===== Unsupervised test report =====
# mkdir -p ${results_dir}
# if [ "$mask_part" -eq 1 ]; then
#     results_dir="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}_mask"
#     unsup_ano_score="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}_mask/unsup_score_mean.csv"
# else
#     results_dir="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}"
#     unsup_ano_score="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}/unsup_score_mean.csv"
# fi
# python3 report_unsup.py \
# --data_version=$dataset_version \
# --results_dir=$results_dir \
# --unsup_ano_score=$unsup_ano_score

# ===== Supervised test report =====
results_dir="../thesis_exp/${dataset_version}/${sup_model_version}/cls"
mkdir -p ${results_dir}
sup_conf_score="../thesis_exp/${dataset_version}/${sup_model_version}/cls/sup_conf.csv"
python3 report_sup.py \
--data_version=$dataset_version \
--results_dir=$results_dir \
--sup_conf_score=$sup_conf_score \

# ===== Combine test report =====
# results_dir="../thesis_exp/${dataset_version}/${sup_model_version}_${model_version}/cls"
# mkdir -p ${results_dir}
# sup_conf_score="../thesis_exp/${dataset_version}/${sup_model_version}/cls/sup_conf.csv"
# if [ "$mask_part" -eq 1 ]; then
#     unsup_ano_score="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}_mask/unsup_score_mean.csv"
# else
#     unsup_ano_score="../thesis_exp/${dataset_version}/${model_version}/cls/${measure}/unsup_score_mean.csv"
# fi
# python3 report_combine.py \
# --data_version=$dataset_version \
# --results_dir=$results_dir \
# --sup_conf_score=$sup_conf_score \
# --unsup_ano_score=$unsup_ano_score
