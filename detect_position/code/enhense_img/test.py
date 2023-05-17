from PIL import Image, ImageEnhance
import os
# add contrast
base_dir = '/home/mura/Mura_ShiftNet/detect_position/typec+b1/resized/actual_pos/bounding_box'
save_dir = './res'
os.makedirs(save_dir, exist_ok=True)

img_list = os.listdir(base_dir)
for fn in img_list:
    img = Image.open(os.path.join(base_dir,fn))
    img = ImageEnhance.Contrast(img).enhance(5)
    img.save(os.path.join(save_dir, f'en_{fn}'))