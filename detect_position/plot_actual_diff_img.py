import pandas as pd
import os
import shutil
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from glob import glob

parser = parser = argparse.ArgumentParser()
parser.add_argument('-dv', '--dataset_version', type=str, default=None, required=True)
parser.add_argument('-mv', '--model_version', type=str, default=None, required=True)
parser.add_argument('-id', '--img_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)

args = parser.parse_args()
dataset_version = args.dataset_version
model_version = args.model_version
img_dir = args.img_dir
save_dir = args.save_dir