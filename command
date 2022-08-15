# use_spectral_norm_D whether to add spectral norm to D, it helps improve results
# which_model_netD selects model to use for netD, [basic|densenet]
# which_model_netG selects model to use for netG [unet_256| unet_shift_triple| \
#                                                 res_unet_shift_triple|patch_soft_unet_shift_triple| \
#                                                 res_patch_soft_unet_shift_triple| face_unet_shift_triple]
# model chooses which model to use. [shiftnet|res_shiftnet|patch_soft_shiftnet|res_patch_soft_shiftnet|test]
# shift_sz shift_sz>1 only for \'soft_shift_patch\'.'
# mask_thred number to decide whether a patch is masked

# =====train=====
# RGB
python train_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--niter=200 --lr=0.0005 --lr_policy=cosine --random_choose_num=10000 --crop_image_num=64 \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/0527_512/train/normal/'

# gray
python train_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=1 --output_nc=1 --color_mode='gray' \
--niter=200 --lr=0.0005 --lr_policy=cosine \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/train/normal/'

# =====test=====
# RGB
python test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MSE' --which_epoch='latest' \
--normal_how_many=13272 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/0527_512/test/normal/' \
--smura_how_many=3940 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/all_smura/'

# --normal_how_many=13272 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/normal/' \
# --smura_how_many=3940 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/all_smura/'

# =====Type-C=====
python test_type_c.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MSE' --which_epoch='latest' \
--normal_how_many=0 --testing_normal_dataroot='' \
--smura_how_many=239 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/typec_4k_8k/4k/'
# --smura_how_many=350 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/typec_4k_8k/8k/'

python ./test_type_c_plus.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' --inpainting_mode='ShiftNet' --measure_mode='MSE_sliding' --which_epoch='250' --testing_normal_dataroot='' --testing_smura_dataroot='E:/CSE/AI/Mura/mura_data/typecplus/'

python ./test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' --inpainting_mode='ShiftNet' --measure_mode='Mask_MSE_sliding' --which_epoch='250' --normal_how_many=31 --testing_normal_dataroot='E:/CSE/AI/Mura/mura_data/d17/test/normal_8k/' --smura_how_many=31 --testing_smura_dataroot='E:/CSE/AI/Mura/mura_data/typecplus/'