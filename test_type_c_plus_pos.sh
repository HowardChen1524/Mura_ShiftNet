#!/bin/bash

python ./test_type_c_plus.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MSE_sliding_typecplus' --which_epoch='200' \
--testing_normal_dataroot='' \
--testing_smura_dataroot='/home/levi/mura_data/typecplus/img/'