import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw

def roc(labels, scores, name):

    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    plot_roc_curve(fpr, tpr, name)
    
    return roc_auc, optimal_th
def plot_roc_curve(fpr, tpr, name):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name+'_roc_curve.png')
    plt.clf()
def plot_distance_distribution(n_scores, s_scores, name):
    # bins = np.linspace(0.000008,0.00005) # Mask MSE
    # n_weights = np.ones_like(n_scores)/float(len(s_scores))
    # s_weights = np.ones_like(s_scores)/float(len(s_scores))
    plt.hist(s_scores, bins=30, alpha=0.5, density=True, label="smura")
    plt.hist(n_scores, bins=30, alpha=0.5, density=True, label="normal")
    plt.xlabel('Anomaly Score')
    plt.title('Distribution')
    plt.legend(loc='upper right')
    plt.savefig(name + '_score.png')
    plt.clf()
def plot_distance_scatter(n_MSE, s_MSE, n_SSIM, s_SSIM, name):
    # normal
    x1 = n_MSE
    y1 = n_SSIM
    # smura
    x2 = s_MSE
    y2 = s_SSIM
    # 設定座標軸
    # normal
    plt.xlabel("MSE")
    plt.ylabel("SSIM")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.legend(loc='upper right')
    plt.savefig(name + '_normal_scatter.png')
    plt.clf()
    # smura
    plt.xlabel("MSE")
    plt.ylabel("SSIM")
    plt.title('scatter')
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(name + '_smura_scatter.png')
    plt.clf()
    # all
    plt.xlabel("MSE")
    plt.ylabel("SSIM")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(name + '_scatter.png')
    plt.clf()
def MSE_SSIM_prediction(labels, MSE_scores, SSIM_scores, name):
    # score = a*MSE + b*SSIM + c
    best_a, best_b, best_c = 0, 0, 0
    best_auc = 0
    for ten_a in range(0, 100, 1):
        a = ten_a/100.0
        for ten_b in range(0, 100, 1):
            b = ten_b/100.0
            for ten_c in range(0, 100, 1):
                c = ten_c/100.0
                scores = a*MSE_scores + b*SSIM_scores + c
                fpr, tpr, th = roc_curve(labels, scores)
                current_auc = auc(fpr, tpr)
                if current_auc >= best_auc:
                    best_auc = current_auc
                    best_a = a
                    best_b = b
                    best_c = c

    print(best_auc)
    print(best_a)
    print(best_b)
    print(best_c)

    best_scores = best_a*MSE_scores + best_b*SSIM_scores + best_c
    pred_labels = [] 
    roc_auc, optimal_th = roc(labels, best_scores, name)
    for score in best_scores:
        if score >= optimal_th:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    print("=====Prediction result=====")
    print("Confusion Matrix (row1: TN,FP | row2: FN,TP):\n", cm)
    print("AUC: ", roc_auc)
    print("TPR-FPR Threshold: ", optimal_th)
    print("Accuracy: ", (TP + TN)/DATA_NUM)
    print("Recall (TPR): ", TP/(TP+FN))
    print("TNR: ", TN/(FP+TN))
    print("PPV: ", TP/(TP+FP))
    print("NPV: ", TN/(FN+TN))
    print("False Alarm Rate (FPR): ", FP/(FP+TN))
    print("Leakage Rate (FNR): ", FN/(FN+TP))
    print("F1-Score: ", f1_score(labels, pred_labels)) # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
             
if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    opt.serial_batches = False
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    data_loader = CreateDataLoader(opt)
    dataset_list = [data_loader['normal'],data_loader['smura']]
    
    model = create_model(opt)

    # 每張大圖的代表小圖
    n_MSE_anomaly_score_log = None
    n_SSIM_anomaly_score_log = None
    s_MSE_anomaly_score_log = None
    s_SSIM_anomaly_score_log = None

    for mode, dataset in enumerate(dataset_list): 
        
        MSE_anomaly_score_log = None
        SSIM_anomaly_score_log = None
        print(f"Mode(0:normal,1:smura): {mode}")
        for i, data in enumerate(dataset):
            print(f"Image num: {i}")
            # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
            bs, ncrops, c, h, w = data['A'].size()
            data['A'] = data['A'].view(-1, c, h, w)
            # print(data['A'].shape)

            bs, ncrops, c, h, w = data['B'].size()
            data['B'] = data['B'].view(-1, c, h, w)
            # print(data['B'].shape)
            
            bs, ncrops, c, h, w = data['M'].size()
            data['M'] = data['M'].view(-1, c, h, w)
            # print(data['M'].shape)

            # 建立 input real_A & real_B
            # it not only sets the input data with mask, but also sets the latent mask.
            model.set_input(data) 
            img_scores = model.test()

            MSE_anomaly_score = np.mean(img_scores['MSE']) # MSE
            SSIM_anomaly_score = np.mean(img_scores['SSIM']) # SSIM
            print(f"MSE Mean: {MSE_anomaly_score}")
            print(f"SSIM Mean: {SSIM_anomaly_score}")

            if i == 0:
                MSE_anomaly_score_log = np.array(MSE_anomaly_score)
                SSIM_anomaly_score_log = np.array(SSIM_anomaly_score)
            else:
                MSE_anomaly_score_log = np.append(MSE_anomaly_score_log, MSE_anomaly_score)
                SSIM_anomaly_score_log = np.append(SSIM_anomaly_score_log, SSIM_anomaly_score)

        if mode == 0:
            n_MSE_anomaly_score_log = MSE_anomaly_score_log.copy() # max
            n_SSIM_anomaly_score_log = SSIM_anomaly_score_log.copy() # mean
        else:
            s_MSE_anomaly_score_log = MSE_anomaly_score_log.copy()
            s_SSIM_anomaly_score_log = SSIM_anomaly_score_log.copy()

    plot_distance_distribution(n_MSE_anomaly_score_log, s_MSE_anomaly_score_log, f"d23_8k_{opt.measure_mode}_MSE")
    plot_distance_distribution(n_SSIM_anomaly_score_log, s_SSIM_anomaly_score_log, f"d23_8k_{opt.measure_mode}_SSIM")
    plot_distance_scatter(n_MSE_anomaly_score_log, s_MSE_anomaly_score_log, 
                            n_SSIM_anomaly_score_log, s_SSIM_anomaly_score_log, f"d23_8k_{opt.measure_mode}_MSE_SSIM_combined")
    
    all_MSE_anomaly_score_log = np.concatenate([n_MSE_anomaly_score_log, s_MSE_anomaly_score_log])
    all_SSIM_anomaly_score_log = np.concatenate([n_SSIM_anomaly_score_log, s_SSIM_anomaly_score_log])
    true_label = [0]*n_MSE_anomaly_score_log.shape[0]+[1]*s_MSE_anomaly_score_log.shape[0]

    print("=====Anomaly Score MSE_SSIM=====")
    MSE_SSIM_prediction(true_label, all_MSE_anomaly_score_log, all_SSIM_anomaly_score_log, f"d23_8k_{opt.measure_mode}_MSE_SSIM_combined")
