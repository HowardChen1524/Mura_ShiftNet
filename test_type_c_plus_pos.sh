#!/bin/bash

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MAE_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Mask_MAE_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MSE_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Mask_MSE_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Discounted_L1_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='SSIM_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Mask_SSIM_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Dis_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Mask_Dis_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Style_VGG16_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Mask_Style_VGG16_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Content_VGG16_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='Mask_Content_VGG16_sliding' --which_epoch='200' \
--testing_normal_dataroot='/hcds_vol/private/howard/mura_data/d23_merge/test/test_normal_8k/' \
--testing_smura_dataroot='/hcds_vol/private/howard/mura_data/typecplus/img/'
