from PIL import Image, ImageDraw
import numpy as np
import cv2

fp = r"E:/CSE/AI/Mura/mura_data/typecplus/6A2D51P21BZZ_20220607093559_0_L050P_resize.png"
img = Image.open(fp)
img = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
img = cv2.resize(img, (512,512), interpolation=cv2.INTER_AREA)
img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

# =====actual mura===== 
actual_pos = (int(1078/3.75), int(399/2.109375), int(1092/3.75), int(426/2.109375))
print(actual_pos)
draw = ImageDraw.Draw(img)  
draw.rectangle(actual_pos, outline ="yellow")

img.save(f"./test.png")
