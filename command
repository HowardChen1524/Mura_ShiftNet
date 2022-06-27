# Now model: sliding

# 0407
# none
python train.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='densenet' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/train_data/normal' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal' --how_many=8261 --results_dir='./results/normal_ori/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect' --how_many=3095 --results_dir='./results/smura_ori/' --loadSize=256

# none2
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal' --how_many=8261 --results_dir='./results/normal_ori_2/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect' --how_many=3095 --results_dir='./results/smura_ori_2/' --loadSize=256

# none3
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal' --how_many=8261 --results_dir='./results/normal_ori_3/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect' --how_many=3095 --results_dir='./results/smura_ori_3/' --loadSize=256

# none4
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal' --how_many=8261 --results_dir='./results/normal_ori_4/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect' --how_many=3095 --results_dir='./results/smura_ori_4/' --loadSize=256

# none5
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal' --how_many=8261 --results_dir='./results/normal_ori_5/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect' --how_many=3095 --results_dir='./results/smura_ori_5/' --loadSize=256

# none test=305
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/normal' --how_many=8261 --results_dir='./results/normal_ori_test_305/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/smura' --how_many=305 --results_dir='./results/smura_ori_test_305/' --loadSize=256

# none test=305
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/normal' --how_many=8261 --results_dir='./results/normal_ori_test_305_2/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/smura' --how_many=305 --results_dir='./results/smura_ori_test_305_2/' --loadSize=256

# none test=305
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/normal' --how_many=8261 --results_dir='./results/normal_ori_test_305_3/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/smura' --how_many=305 --results_dir='./results/smura_ori_test_305_3/' --loadSize=256

# sliding crop
python train.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/train_data/normal'
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/normal' --how_many=8261 --results_dir='./results/normal_resizefirst/'
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/defect' --how_many=3095 --results_dir='./results/smura_resizefirst/'

# sliding crop 2
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/normal' --how_many=8261 --results_dir='./results/normal_resizefirst_2/'
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/defect' --how_many=3095 --results_dir='./results/smura_resizefirst_2/'

# sliding crop 3
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/normal' --how_many=8261 --results_dir='./results/normal_resizefirst_3/'
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/defect' --how_many=3095 --results_dir='./results/smura_resizefirst_3/'

# sliding crop 4
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/normal' --how_many=8261 --results_dir='./results/normal_resizefirst_4/'
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/defect' --how_many=3095 --results_dir='./results/smura_resizefirst_4/'

# sliding crop 5
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/normal' --how_many=8261 --results_dir='./results/normal_resizefirst_5/'
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_sliding_256_0604_resizefirst/test_data/defect' --how_many=3095 --results_dir='./results/smura_resizefirst_5/'

#0429/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test/normal
python train.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/train/normal' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_345_0429/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_345_0429/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_345_0429_2/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_345_0429_2/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_345_0429_3/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_345_0429_3/' --loadSize=256

# epoch 60
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_345_0429_epoch_60/' --loadSize=256 --which_epoch=60
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_345_0429_epoch_60/' --loadSize=256 --which_epoch=60

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_345_0429_epoch_60_2/' --loadSize=256 --which_epoch=60
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_345_0429_epoch_60_2/' --loadSize=256 --which_epoch=60

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_345_0429_epoch_60_3/' --loadSize=256 --which_epoch=60
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_345_0429_epoch_60_3/' --loadSize=256 --which_epoch=60

#train d12 test d15
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_train_d12_test_d15/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_train_d12_test_d15/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_train_d12_test_d15_2/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_train_d12_test_d15_2/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_train_d12_test_d15_3/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_train_d12_test_d15_3/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_train_d12_test_d15_4/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_train_d12_test_d15_4/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/normal' --how_many=10146 --results_dir='./results/normal_ori_train_d12_test_d15_5/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220429/20220429_merge/test/smura' --how_many=345 --results_dir='./results/smura_ori_train_d12_test_d15_5/' --loadSize=256

#train d12 test d17
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/normal' --how_many=13272 --results_dir='./results/normal_ori_train_d12_test_d17/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/smura' --how_many=386 --results_dir='./results/smura_ori_train_d12_test_d17/' --loadSize=256

python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/normal' --how_many=13272 --results_dir='./results/normal_ori_train_d12_test_d17_2/' --loadSize=256
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1 --dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220527_merge/test/smura' --how_many=386 --results_dir='./results/smura_ori_train_d12_test_d17_2/' --loadSize=256

# train
python train.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 \
--dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/train_data/normal'

# test 
python test.py --batchSize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --which_model_netG='unet_shift_triple' --model='shiftnet' \
--shift_sz=1 --mask_thred=1 --loadSize=256 --inpainting_mode='ShiftNet' --measure_mode='MSE' --which_epoch='50' \
--normal_how_many=8261 --testing_normal_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/normal/' \
--smura_how_many=3095 --testing_smura_dataroot='/home/levi/Howard/Mura/mura_data/RGB/20220407/20220407_merged/test_data/defect/' \
