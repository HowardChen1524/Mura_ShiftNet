import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# 將指定圖中的每張小圖csv排序
# def enhance_img(img,factor=5):
#   enh_con = ImageEnhance.Contrast(img)
#   new_img = enh_con.enhance(factor=factor)
#   return new_img

# for mode in [0, 1]:
#   spec_img_path = f'/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/ShiftNet_SSIM_d23_4k/mura_d23_4k_typec/typec4k/{mode}/'
#   spec_img_list = [fn for fn in os.listdir(spec_img_path)]
#   print(spec_img_list)
#   img_path = f'/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/ShiftNet_SSIM_d23_4k/mura_d23_4k_typec/MSE_sliding/check_inpaint/{mode}/'
#   for fn in os.listdir(img_path):
#       if fn in spec_img_list:
#           pil_img = Image.open(os.path.join(spec_img_path, fn))
#           pil_img_en = enhance_img(pil_img)
#           pil_img_en.save(os.path.join(spec_img_path, f'en_{fn}'))
#           df = pd.read_csv(os.path.join(img_path, f'{fn}/pos_score.csv'))
#           df = df.sort_values(by='score', ascending=False)
#           df.to_csv(os.path.join(spec_img_path, f'{fn}_pos_score_sorted.csv'))


# def plot_score_distribution(n_scores, s_scores):
#     plt.clf()
#     # plt.xlim(4e-05, 4e-04)
#     # plt.xlim(5e-05, 1.5e-04)
#     # plt.xlim(1e-05, 3.5e-05)
#     plt.hist(n_scores, bins=50, alpha=0.5, density=True, label="normal")
#     plt.hist(s_scores, bins=50, alpha=0.5, density=True, label="smura")
#     plt.xlabel('Anomaly Score')
#     plt.title('Score Distribution')
#     plt.legend(loc='upper right')
#     plt.savefig(f"./typec_max_dist.png")
#     plt.clf()

# # df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/ShiftNet_SSIM_d23_4k/mura_d23_4k/MSE_sliding/unsup_score_max.csv')
# df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/ShiftNet_SSIM_d23_4k/mura_d23_4k_typec/MSE_sliding/unsup_score_max.csv')

# normal_filter = df.label == 0
# smura_filter = df.label == 1
# outlier_filter = df.score_max > 1e-04
# nscore = df[normal_filter & outlier_filter].score_max.to_numpy() # 168
# sscore = df[smura_filter & outlier_filter].score_max.to_numpy() # 13, typec 30
# print(nscore.shape)
# print(sscore.shape)
# plot_score_distribution(nscore, sscore)
# df[smura_filter & outlier_filter].to_csv('./typec_smura_outlier.csv')

# df1 = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/temp/sup_conf_remove.csv')
# df2 = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/temp/sup_conf_no_remove.csv')
# df = df1.merge(df2, left_on='name', right_on='name')
# print((df.conf_x - df.conf_y).sort_values().to_numpy())

img = np.array(Image.open('/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/16/ori_diff_patches/4C2D34N6XAZZ_20220626001612_0_L050P_resize.png/real/en_0.png'))
s_img = Image.fromarray(img[1:63,1:63,:])
s_img.save('test.png')