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
parser.add_argument('-nc', '--n_cluster', type=int, required=True)

def plot_3D_dist(X_n, X_s, name):
    # 建立 3D 圖形
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # 繪製 3D 座標點
    ax.scatter(X_n['B_mean'], X_n['G_mean'], X_n['R_mean'], cmap='Blues', marker='o', label='Normal')
    ax.scatter(X_s['B_mean'], X_s['G_mean'], X_s['R_mean'], cmap='Reds', marker='o', label='Smura')

    # ax.set_xlim([0, 255])
    # ax.set_ylim([0, 255])
    # ax.set_zlim([0, 255])

    # 顯示圖例
    ax.legend()

    # 顯示圖形
    plt.savefig(f'./{name}')

def join_path(p1, p2):
    return os.path.join(p1, p2)

if __name__ == "__main__":

    args = parser.parse_args()
    n_cluster = args.n_cluster

    # read X_n
    X_n = pd.read_csv('X_n.csv')
    # X_s = pd.read_csv(f'X_s.csv')

    X_s = pd.read_csv(f'X_s_cluster_{n_cluster}.csv')


    plt.hist(X_n['B_mean'], bins=100, range=(0,255), density=True, alpha=0.5, label='Normal')
    plt.hist(X_s['B_mean'], bins=50, range=(0,255), density=True, alpha=0.5, label='Smura')
    plt.xlabel('Pixel Value')
    plt.title('B Channel Distribution')
    plt.legend(loc="upper right")
    # plt.savefig(f'./hist_B_mean.png')
    plt.savefig(f'./cluster_{n_cluster}_hist_B_mean.png')
    plt.clf()

    plt.hist(X_n['G_mean'], bins=100, range=(0,255), density=True, alpha=0.5, label='Normal')
    plt.hist(X_s['G_mean'], bins=50, range=(0,255), density=True, alpha=0.5, label='Smura')
    plt.xlabel('Pixel Value')
    plt.title('G Channel Distribution')
    plt.legend(loc="upper right")
    # plt.savefig(f'./hist_G_mean.png')
    plt.savefig(f'./cluster_{n_cluster}_hist_G_mean.png')
    plt.clf()

    plt.hist(X_n['R_mean'], bins=100, range=(0,255), density=True, alpha=0.5, label='Normal')
    plt.hist(X_s['R_mean'], bins=50, range=(0,255), density=True, alpha=0.5, label='Smura')
    plt.xlabel('Pixel Value')
    plt.title('R Channel Distribution')
    plt.legend(loc="upper right")
    # plt.savefig(f'./hist_R_mean.png')
    plt.savefig(f'./cluster_{n_cluster}_hist_R_mean.png')
    plt.clf()

    