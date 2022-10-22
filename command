# use_spectral_norm_D whether to add spectral norm to D, it helps improve results
# which_model_netD selects model to use for netD, [basic|densenet]
# which_model_netG selects model to use for netG [unet_256| unet_shift_triple| /
#                                                 res_unet_shift_triple|patch_soft_unet_shift_triple| /
#                                                 res_patch_soft_unet_shift_triple| face_unet_shift_triple]
# model chooses which model to use. [shiftnet|res_shiftnet|patch_soft_shiftnet|res_patch_soft_shiftnet|test]
# shift_sz shift_sz>1 only for /"soft_shift_patch/"."
# mask_thred number to decide whether a patch is masked

# =====train=====

# =====test=====
python ./test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="R_square" --which_epoch="200" --testing_normal_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/normal_8k/" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/smura_8k/" --nThread=0

python ./test_sliding_normalized_rsquare.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="MSE_sliding" --which_epoch="200" --testing_normal_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/normal_8k/" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/smura_8k/" --nThread=0

python ./test_sliding_MSE_SSIM.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --crop_stride=32 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="Mask_MSE_SSIM_sliding" --which_epoch="200" --testing_normal_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/normal_8k/" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/smura_8k/" --nThread=0

# =====Type-C=====
python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="MSE_sliding" --which_epoch="200" --testing_normal_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/normal_8k/" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/typecplus/" --nThread=0

python ./test_type_c_plus.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="MSE_sliding" --which_epoch="200" --testing_normal_dataroot="" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/typecplus/" --nThread=0

python ./test_type_c_plus_normalized.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="MSE_sliding" --which_epoch="200" --testing_normal_dataroot="E:/CSE/AI/Mura/mura_data/d23/test/normal_8k/" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/typecplus/" --nThread=0

python ./test_type_c_plus.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD="basic" --which_model_netG="unet_shift_triple" --model="shiftnet" --shift_sz=1 --mask_thred=1 --loadSize=64 --fineSize=64 --overlap=0 --dataset_mode="aligned_sliding" --mask_type="center" --input_nc=3 --output_nc=3 --color_mode="RGB" --inpainting_mode="ShiftNet" --measure_mode="Mask_MSE_sliding" --which_epoch="200" --testing_normal_dataroot="" --testing_smura_dataroot="E:/CSE/AI/Mura/mura_data/typecplus/" --nThread=0

