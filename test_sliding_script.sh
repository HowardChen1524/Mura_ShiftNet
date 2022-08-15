#!/bin/bash

python ./test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 /
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' /
--inpainting_mode='ShiftNet' --measure_mode='MSE_sliding' --which_epoch='250' /
--normal_how_many=31 --testing_normal_dataroot='E:/CSE/AI/Mura/mura_data/d17/test/normal_8k/' /
--smura_how_many=31 --testing_smura_dataroot='E:/CSE/AI/Mura/mura_data/typecplus/'




# --normal_how_many=13272 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/0527_512/test/normal/'
# --smura_how_many=386 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/0527_512/testdata/smura/'
# --smura_how_many=3940 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/0527_512/test/allsmura/'