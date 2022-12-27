#!/bin/bash
# /hcds_vol/private/howard/mura_data/typecplus/img/

declare -a measure_list=(
                        #  "MSE_sliding"
                         "Mask_MSE_sliding"
                        #  "SSIM_sliding"
                        #  "Mask_SSIM_sliding"
                        #  "Dis_sliding"
                        #  "Mask_Dis_sliding"
                        #  "Style_VGG16_sliding"
                        #  "Mask_Style_VGG16_sliding"
                        #  "Content_VGG16_sliding"
                        #  "Mask_Content_VGG16_sliding"
                        )
              
declare -a sup_model_list=(
    '/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_0.pt'
    '/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_1.pt'
    '/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_2.pt'
)
# sup_model='/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_0.pt'
# sup_model='/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_1.pt'
# sup_model='/home/ldap/sallylin/Howard/Mura_ShiftNet/log/Supervised/ensemble_d23/model_2.pt'

sup_model_version="ensemble_d23"

# model_version="ShiftNet_SSIM_d23_4k"
# model_version="ShiftNet_SSIM_d23_8k"
# model_version="ShiftNet_d23_8k"

model_version="PEN-NET_d23_8k"

# dataset_version="mura_d23_4k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# sup_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_4k/" # for unsupervised model
# normal_num=5
# smura_num=5
# normal_num=16295
# smura_num=181

# dataset_version="mura_d23_4k_typec"
# sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# sup_csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_4k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/typec_4k_8k/4k/" # for unsupervised model
# normal_num=239
# smura_num=239

# dataset_version="d23_8k_typecplus"
# sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/typecplus/img/" # for unsupervised model
# normal_num=1
# smura_num=31

# d23 test
# dataset_version="d23_8k"
# sup_data_path="/hcds_vol/private/howard/mura_data/d23_merge/" # for supervised model
# csv_path="/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv" # for supervised model
# unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/" # for unsupervised model
# unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d23_merge/test/test_smura_8k/" # for unsupervised model
# normal_num=541
# smura_num=143
# conf_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d23_8k'
# # score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/ShiftNet_SSIM_d23_8k/d23_8k/Mask_MSE_sliding'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/PEN-Net_d23_8k/d23_8k/Mask_MSE_sliding'

# d24 d25 blind test
dataset_version="d24_d25_8k"
sup_data_path="/hcds_vol/private/howard/mura_data/d25_merge/"
# csv_path="/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv"
csv_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d25_data_merged_new.csv"
unsup_test_normal_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_normal_8k/"
unsup_test_smura_path="/hcds_vol/private/howard/mura_data/d25_merge_8k/d2425_all_smura_8k/"
normal_num=87
smura_num=85
conf_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k'
# score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/ShiftNet_SSIM_d23_8k/d24_d25_8k/Mask_MSE_sliding'
score_csv_dir='/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Unsupervised/PEN-Net_d23_8k/d24_d25_8k/Mask_MSE_sliding'

# only supervised model
# for sup_model in ${sup_model_list[@]}
# do
#     python3 sup_gen_res.py \
#     --sup_model_version=$sup_model_version --data_version=$dataset_version \
#     --sup_model_path=$sup_model --data_dir=$sup_data_path --csv_path=$csv_path \
#     --gpu_id=0
# done

# only unsupervised model
# for measure in ${measure_list[@]}
# do
#     python3 test_sliding.py \
#     --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 \
#     --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" \
#     --inpainting_mode="ShiftNet" --measure_mode=$measure --checkpoints_dir='./log/Unsupervised' --results_dir='./exp_result/Unsupervised' --model_version=$model_version --which_epoch="200" \
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
#     --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" \
#     --inpainting_mode="ShiftNet" --measure_mode=$measure --checkpoints_dir='./log' --model_version=$model_version --which_epoch="400" \
#     --data_version $dataset_version \
#     --normal_how_many=$normal_num --testing_normal_dataroot=$unsup_test_normal_path \
#     --smura_how_many=$smura_num --testing_smura_dataroot=$unsup_test_smura_path \
#     --sup_model_path=$sup_model --data_dir=$sup_data_path --csv_path=$csv_path --gpu_ids=0
# done

# find line
for measure in ${measure_list[@]}
do
    python3 sup_unsup_find_th_or_test.py \
    --measure_mode=$measure --model_version=$model_version --data_version $dataset_version \
    --conf_csv_dir=$conf_csv_dir \
    --score_csv_dir=$score_csv_dir \
    --checkpoints_dir='./log' --results_dir='./exp_result/Ensemble' --using_threshold
done
