from PIL import Image
import numpy as np
import os
dir1 = '/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/16/union/0.0125_diff_pos_area_0/imgs'
dir2 = '/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/32/union/0.0125_diff_pos_area_0/imgs'

area_img1_list = []
area_img2_list = []
for fn in os.listdir(dir1):
        img1 = np.array(Image.open(os.path.join(dir1,fn)))/255
        img2 = np.array(Image.open(os.path.join(dir2,fn)))/255
        area_img1 = np.sum(img1[img1==1])
        area_img2 = np.sum(img2[img2==1])
        # print(area_img1)
        # print(area_img2)
        area_img1_list.append(area_img1)
        area_img2_list.append(area_img2)
print(np.sum(area_img1))
print(np.sum(area_img2))
print(np.sum(area_img1)/np.sum(area_img2))