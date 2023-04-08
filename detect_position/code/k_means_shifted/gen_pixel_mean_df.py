from sklearn.cluster import KMeans
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import math
import cv2
import os
import joblib


parser = argparse.ArgumentParser()
parser.add_argument('-nd', '--normal_dir', type=str, default=None, required=True)
parser.add_argument('-md', '--smura_dir', type=str, default=None, required=True)
parser.add_argument('-sd', '--save_dir', type=str, default=None, required=True)
parser.add_argument('-nc', '--n_clusters', type=int, required=True)

def join_path(p1, p2):
    return os.path.join(p1, p2)

if __name__ == "__main__":

    args = parser.parse_args()
    normal_dir = args.normal_dir
    smura_dir = args.smura_dir
    save_dir = args.save_dir
    n_clusters = args.n_clusters
    os.makedirs(save_dir, exist_ok=True)

    n_blue_means = []
    n_green_means = []
    n_red_means = []
    
    for idx, fn in enumerate(os.listdir(normal_dir)):
        print(f'd23 4k num: {idx+1}, {fn}')    
        img = cv2.imread(join_path(normal_dir, fn))
        B, G, R = cv2.split(img)
        
        blue_mean = np.mean(B)
        green_mean = np.mean(G)
        red_mean = np.mean(R)

        n_blue_means.append(blue_mean)
        n_green_means.append(green_mean)
        n_red_means.append(red_mean)

    X_n = pd.DataFrame(list(zip(n_blue_means, n_green_means, n_red_means)), columns=['B_mean', 'G_mean', 'R_mean'])
    
    # save X_n
    X_n.to_csv('X_n.csv', index=False)

    s_blue_means = []
    s_green_means = []
    s_red_means = []
    
    for idx, fn in enumerate(os.listdir(smura_dir)):
        print(f'd23 4k num: {idx+1}, {fn}')    
        img = cv2.imread(join_path(smura_dir, fn))
        B, G, R = cv2.split(img)
        
        blue_mean = np.mean(B)
        green_mean = np.mean(G)
        red_mean = np.mean(R)

        s_blue_means.append(blue_mean)
        s_green_means.append(green_mean)
        s_red_means.append(red_mean)

    X_s = pd.DataFrame(list(zip(s_blue_means, s_green_means, s_red_means)), columns=['B_mean', 'G_mean', 'R_mean'])
    X_s.to_csv(f'X_s_cluster_{n_clusters}.csv', index=False)
    