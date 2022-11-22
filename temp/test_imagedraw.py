from PIL import Image, ImageDraw
import pandas as pd

# =====actual mura=====
# if 4k or 8k
# ...

df = pd.read_csv('../MURA_XY.csv')
actual_img_path = df[df['PIC_ID'] == data['A_paths'][?:?]]

# read img and resize 512,512
actual_img = Image.open(actual_img_path).convert(self.opt.color_mode)
# if not 512,512 -> resize
ORISIZE = 512
if actual_img.size != (ORISIZE, ORISIZE):
    actual_img = actual_img.resize((ORISIZE, ORISIZE), Image.BICUBIC)

# create bounding_box 108*100
bounding_box_x = int(200 / 3.75) + 1 # +1 是為了使邊長為偶數
bounding_box_y = int(200 / 2)

# 中心點位置 - bounding box 邊長一半 （依照 1920*1080 -> 512*512 比例縮放）
# 4k
crop_x0 = int(actual_img['X'] / 2 / 3.75) - (bounding_box_x / 2)
crop_y0 = int(actual_img['Y'] / 2 / 2) - (bounding_box_y / 2)
crop_x1 = int(actual_img['X'] / 2 / 3.75) + (bounding_box_x / 2)
crop_y1 = int(actual_img['Y'] / 2 / 2) + (bounding_box_y / 2)

# 8k
crop_x0 = int(actual_img['X'] / 4 / 3.75) - (bounding_box_x / 2)
crop_y0 = int(actual_img['Y'] / 4 / 2) - (bounding_box_y / 2)
crop_x1 = int(actual_img['X'] / 4 / 3.75) + (bounding_box_x / 2)
crop_y1 = int(actual_img['Y'] / 4 / 2) + (bounding_box_y / 2)

# if 超出邊界
# ...

pos = [(crop_x0, crop_y0), (crop_x1, crop_y1)]

actual_img = ImageDraw.Draw(actual_img)  
actual_img.rectangle(actual_img, outline ="red")
actual_img.save('test.png')

# =====predict mura=====
h, w = 512, 512
stride = 32

max_crop_img_pos = 175

# [(x0, y0), (x1, y1)]
x = max_crop_img_pos % 16
y = max_crop_img_pos // 16

crop_x = x*stride - 1
crop_y = y*stride - 1

# print(crop_x)
# print(crop_y)

if crop_x < 0:
    crop_x = 0
if crop_y < 0:
    crop_y = 0

if x == 15:
    crop_x = w - 64 - 1
if y == 15:
    crop_y = h - 64 - 1

# print(crop_x)
# print(crop_y)

pos = [(crop_x, crop_y), (crop_x+64, crop_y+64)]

# creating new Image object
img = Image.new("RGB", (h, w))

# create rectangle image
img = ImageDraw.Draw(img)  
img.rectangle(pos, outline ="red")
img.save('test.png')