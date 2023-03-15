from PIL import Image, ImageDraw
import numpy as np

img_path = '/home/sallylab/Howard/Mura_ShiftNet/detect_position/typec+b1/no_padding_edge/4-connected/img_contain_gt/0.0150_diff_pos_area_40_add_gt/6A2D5190QBZZ_20220605152552_0_L050P_resize.png'
pos_list = [438]
img = np.array(Image.open(img_path).convert('L'))
print(img.shape)
# pad_width = ((14,14), (14,14))
# img = np.pad(img, pad_width, mode='constant', constant_values=0)
img = Image.fromarray(img)
for pos in pos_list:
    y = pos//30
    x = pos%30
    x0 = x*16
    y0 = y*16
    x1 = x*16+64
    y1 = y*16+64
    print(x0,y0,x1,y1)
    draw = ImageDraw.Draw(img)  
    draw.rectangle(((x0,y0),(x1,y1)), outline ="red")

img.save('res.png')
