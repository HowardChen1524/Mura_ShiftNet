from PIL import Image, ImageEnhance
import os
# add contrast
base_dir = '/home/mura/mura_data/typed_demura/test_smura_wo_label'
save_dir = './res'
os.makedirs(save_dir, exist_ok=True)

img_list = os.listdir(base_dir)
for fn in img_list:
    img = Image.open(os.path.join(base_dir,fn))
    img = ImageEnhance.Contrast(img).enhance(5)
    img.save(os.path.join(save_dir, f'en_{fn}'))