#!/bin/bash

path_csv='/home/levi/mura_data/d23/data_merged.csv'
path_normal='/home/levi/mura_data/d23/test/smura/'
path_save_4k=''
path_save_8k='/home/levi/mura_data/d23/test/smura_8k/'

python split_4k_8k.py --path_csv=$path_csv --path_normal=$path_normal --path_save_8k=$path_save_8k