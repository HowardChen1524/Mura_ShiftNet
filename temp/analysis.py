import os
import sys
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
# df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/temp/mura_d23_8k_with_sup/mura_d24_d25_8k_ShiftNet_SEResNeXt101/Mask_MSE_sliding/unsup_score.csv')
# normal_filter = df.label == 0
# outlier_filter = df.score <= 5.5e-05
# print(df[normal_filter & outlier_filter])
def enhance_img(img,factor=5):
  enh_con = ImageEnhance.Contrast(img)
  new_img = enh_con.enhance(factor=factor)
  return new_img



for mode in [0, 1]:
  spec_img_path = f'/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/ShiftNet_SSIM_d23_4k/mura_d23_4k_typec/typec4k/{mode}/'
  spec_img_list = [fn for fn in os.listdir(spec_img_path)]
  print(spec_img_list)
  img_path = f'/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/ShiftNet_SSIM_d23_4k/mura_d23_4k_typec/MSE_sliding/check_inpaint/{mode}/'
  for fn in os.listdir(img_path):
      if fn in spec_img_list:
          pil_img = Image.open(os.path.join(spec_img_path, fn))
          pil_img_en = enhance_img(pil_img)
          pil_img_en.save(os.path.join(spec_img_path, f'en_{fn}'))
          df = pd.read_csv(os.path.join(img_path, f'{fn}/pos_score.csv'))
          df = df.sort_values(by='score', ascending=False)
          df.to_csv(os.path.join(spec_img_path, f'{fn}_pos_score_sorted.csv'))
        
# pd.read_csv()