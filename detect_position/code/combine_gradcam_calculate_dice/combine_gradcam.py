import os
from collections import defaultdict
from PIL import Image
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-dd', '--data_dir', type=str, default=None, required=True)
parser.add_argument('-cs', '--crop_stride', type=int, default=None, required=True)
parser.add_argument('-th', '--threshold', type=float, default=None, required=True)
parser.add_argument('-ma', '--min_area', type=int, default=None, required=True)
parser.add_argument('-sd', '--sup_data_dir', type=int, default=None, required=True)

args = parser.parser_args()
dataset_version = args.dataset_version
data_dir = args.data_dir
crop_stride = args.crop_stride
th = args.threshold
min_area = args.min_area
sup_data_dir = args.sup_data_dir

def join_path(path1, path2):
    return (os.path.join(path1,path2))

def dice_coefficient(img1, img2):
    # Ensure the images have the same shape
    assert img1.shape == img2.shape, "Error: Images have different shapes"
    # Calculate the Dice coefficient
    # Calculate the intersection
    intersection = np.sum(img1 * img2)
    total_white_pixel = np.sum(img1) + np.sum(img2)

    dice = (2 * intersection) / total_white_pixel
    return dice

gt_dir = join_path(args.data_dir, f'{dataset_version}/actual_pos/ground_truth')
unsup_dir = join_path(args.data_dir, f'{dataset_version}/{crop_stride}/union/{th:.4f}_diff_pos_area_{min_area}')
sup_dir = join_path(args.data_dir, f'{dataset_version}/sup_gradcam')
save_dir = join_path(args.data_dir, f'{dataset_version}/sup_unsup_combined')

row_data = defaultdict(float)
dice_list = []
for fn in os.listdir(gt_dir):
    gt_img = np.array(Image.open(join_path(gt_dir,fn)))/255
    sup_img = np.array(Image.open(join_path(sup_dir,fn)))/255
    unsup_img = np.array(Image.open(join_path(unsup_dir,f'imgs/{fn}')))/255
    combine_img = sup_img * unsup_img
    print(combine_img)
    combine_img = Image.fromarray(combine_img)
    combine_img.save(join_path(save_dir,fn))
    dice = dice_coefficient(gt_img, combine_img)
    dice_list.append(dice)
    row_data[fn] = dice
    print(f"{fn}: {dice}")

df = pd.DataFrame(data=list(row_data.items()),columns=['fn','dice'])
print(df['dice'].mean())
df.to_csv(join_path(save_dir,f'dice.csv'),index=False)
nonzero_df = df[df['dice']!=0]
print(nonzero_df['dice'].mean())
nonzero_df.to_csv(join_path(save_dir,f"dice_nonzero.csv"),index=False)
with open (join_path(save_dir,f"dice_mean.txt"), 'w') as f:
    msg  = f"Typec+b1\ndice mean：{df['dice'].mean()}\n"
    msg += f"dice mean：{nonzero_df['dice'].mean()} (不包含dice=0)\n"
    msg += f"命中張數(至少一處)：{df[df['dice']!=0].shape[0]}\n"
    msg += f"命中率(至少一處)：{nonzero_df.shape[0]/df.shape[0]}"
    f.write(msg)
    

