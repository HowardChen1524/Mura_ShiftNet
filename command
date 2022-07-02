# Now model: 0407 ori aligned resize 50

# train
python train.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 --fineSize=256 --dataset_mode='aligned_resized' --display_port=9999 \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/train_data/normal'

# test 
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 --fineSize=256 --inpainting_mode='ShiftNet' --measure_mode='MSE' --dataset_mode='aligned_resized' --which_epoch='50' \
--normal_how_many=8261 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal/' \
--smura_how_many=3095 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect/'

# coding test
python train_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 --fineSize=256 --dataset_mode='aligned_sliding' --mask_type='center' --mask_sub_type='rect' --display_port=9999 --random_choose_num=5000 \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/train_data/normal' --continue_train

# sliding test
python test_sliding.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 --fineSize=256 --inpainting_mode='ShiftNet' --measure_mode='MSE' --dataset_mode='aligned_sliding' --mask_type='center' --mask_sub_type='rect' --which_epoch='20' \
--normal_how_many=8261 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal/' \
--smura_how_many=3095 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect/'

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 --fineSize=256 --inpainting_mode='ShiftNet' --measure_mode='MSE' --dataset_mode='aligned' --which_epoch='latest' \
--normal_how_many=8261 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal/' \
--smura_how_many=3095 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect/'
