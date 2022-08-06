#!/bin/bash

python ./test_type_c_wei.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_combined' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MSE' --which_epoch='100' \
--testing_normal_dataroot='/home/sally/0527_512/testdata/normal_8k/' \
--testing_smura_dataroot='/home/sally/typec_4k_8k/8k/'