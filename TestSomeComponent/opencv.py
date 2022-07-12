import cv2
import numpy as np
from PIL import Image

img = Image.open('./opencv_img.png').convert('L')
img = np.array(img).astype('float32')
print(img.shape)
mask = Image.open('./opencv_mask.png').convert('L')
mask = np.array(mask).astype('uint8')
print(mask.shape)
fake_B = cv2.inpaint(img, mask, 1, cv2.INPAINT_NS)
print(fake_B)
# print(img)
# print(np.where(mask==255.0))
# print(mask[np.where(mask==255.0)].reshape(32,32))