# train
# RGB
python train_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--niter=200 --lr=0.0005 --lr_policy=cosine --display_port=9999 \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/0407_512/train/normal'
# gray
python train_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=1 --output_nc=1 --color_mode='gray' \
--niter=200 --lr=0.0005 --lr_policy=cosine --display_port=9999 \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/0407_512/train/normal'

# test 
# RGB
python test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=3 --output_nc=3 --color_mode='RGB' \
--inpainting_mode='ShiftNet' --measure_mode='MSE' --which_epoch='latest' \
--normal_how_many=8261 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/0407_512/test/normal' \
--smura_how_many=3095 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/0407_512/test/smura'
# gray
python test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 \
--loadSize=64 --fineSize=64 --overlap=0 --dataset_mode='aligned_sliding' --mask_type='center' --input_nc=1 --output_nc=1 --color_mode='gray' \
--inpainting_mode='ShiftNet' --measure_mode='MSE' --which_epoch='latest' \
--normal_how_many=8261 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/0407_512/test/normal' \
--smura_how_many=3095 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/0407_512/test/smura'