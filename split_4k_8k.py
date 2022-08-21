import numpy as np 
import pandas as pd 
import os
from glob import glob
import shutil
import argparse

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

parser = argparse.ArgumentParser()
parser.add_argument("--path_csv", type=str)
parser.add_argument("--path_normal", type=str)
parser.add_argument("--path_save_4k", type=str)
parser.add_argument("--path_save_8k", type=str)

args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.path_csv)
    paths = glob(f"{args.path_normal}*png")

    # mkdirs([path_save_4k,path_save_8k])

    mkdirs([args.path_save_8k]) # only 8k

    for i, path in enumerate(paths):
        print(f"img {i}")
        fn = path[len(args.path_normal):]
        fn_series = df[df['name']==fn]
        # if fn_series['PRODUCT_CODE'].values[0] == 'T850QVN03': # 4k
        #     source = path
        #     dest = args.path_save_4k + fn
        #     if os.path.isfile(source):
        #         shutil.copy(source, dest)
        if fn_series['PRODUCT_CODE'].values[0] == 'T850MVR05': # 8k
            source = path
            dest = args.path_save_8k + fn
            if os.path.isfile(source):
                shutil.copy(source, dest)
