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


# n_clusters = 5

#     normal_dir = "/home/sallylab/min/d23_merge/test/test_normal_4k"
#     typed_dir = "/home/sallylab/min/typed/img"
#     save_dir = f"/home/sallylab/min/typed_shifted_{n_clusters}/img"

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

def find_nearest_point(x, y, z, centers):
    
    distances = [math.sqrt((x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2) for c in centers]
    
    nearest_index = distances.index(min(distances))

    return nearest_index

def join_path(p1, p2):
    return os.path.join(p1, p2)

if __name__ == "__main__":

    args = parser.parse_args()
    normal_dir = args.normal_dir
    smura_dir = args.smura_dir
    save_dir = args.save_dir
    n_clusters = args.n_clusters
    os.makedirs(save_dir, exist_ok=True)

    # 計算所有圖片RGB channel的平均值
    # n_blue_means = []
    # n_green_means = []
    # n_red_means = []
    
    # for idx, fn in enumerate(os.listdir(normal_dir)):
    #     print(f'd23 4k num: {idx+1}, {fn}')    
    #     img = cv2.imread(join_path(normal_dir, fn))
    #     B, G, R = cv2.split(img)
        
    #     blue_mean = np.mean(B)
    #     green_mean = np.mean(G)
    #     red_mean = np.mean(R)

    #     n_blue_means.append(blue_mean)
    #     n_green_means.append(green_mean)
    #     n_red_means.append(red_mean)

    # X_n = pd.DataFrame(list(zip(n_blue_means, n_green_means, n_red_means)), columns=['B_mean', 'G_mean', 'R_mean'])
    
    # # temp save X_n
    # X_n.to_csv('X_n.csv', index=False)
    # X_n = X_n.values

    # read X_n
    X_n = pd.read_csv('X_n.csv')
    X_n = X_n.values
    
    # 分群找出中心點
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_n)
    # joblib.dump(kmeans, f'./d23_4k_{n_clusters}_clusters.pkl')
    centers = kmeans.cluster_centers_
    for idx, c in enumerate(centers):
        print(f"Center{idx+1}: {c}")

    # typed 圖片做平移
    for idx, fn in enumerate(os.listdir(smura_dir)):
        print(f'typed num{idx+1}: {fn}')
        img = cv2.imread(join_path(smura_dir, fn))
        B, G, R = cv2.split(img)

        blue_mean = np.mean(B)
        green_mean = np.mean(G)
        red_mean = np.mean(R)

        # 看 mean 跟哪個 feature 比較近
        pred_class = kmeans.predict(np.array([blue_mean, green_mean, red_mean]).reshape(-1, 3))[0]
        # print(pred_class)
        
        # nearest_idx = find_nearest_point(blue_mean, green_mean, red_mean, centers)
        # print(nearest_idx)
        print(B)
        print(blue_mean)
        
        b_mean_shift = blue_mean - centers[pred_class][0]        
        g_mean_shift = green_mean - centers[pred_class][1]
        r_mean_shift = red_mean - centers[pred_class][2]
        
        print(centers[pred_class][0])
        print(b_mean_shift)

        
        B = B - b_mean_shift
        G = G - g_mean_shift
        R = R - r_mean_shift
        print(B)
        print(np.mean(B))
        raise
        img = cv2.merge([B, G, R])
        cv2.imwrite(join_path(save_dir, fn), img)

    # plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
    # plt.savefig(f'./d23_4k_{n_clusters}_clusters')