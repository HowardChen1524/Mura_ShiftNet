import torch
import lpips
from IPython import embed

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix

use_gpu = False         # Whether to use GPU
spatial = False         # Return a spatial map of perceptual distance.
# Linearly calibrated models (LPIPS)
loss_fn = lpips.LPIPS(net='alex', spatial=spatial) # Can also set net = 'squeeze' or 'vgg'
# loss_fn = lpips.LPIPS(net='alex', spatial=spatial, lpips=False) # Can also set net = 'squeeze' or 'vgg'
if(use_gpu):
    loss_fn.cuda()

def roc(labels, scores, name_model):
    
    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]
    
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_th

def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name_model+'_roc_curve.png')
    plt.show()
    plt.clf()

def prediction(labels, scores, name_model):
    pred_labels = [] 

    roc_auc, optimal_th = roc(labels, scores, name_model)
    
    for score in scores:
        if score >= optimal_th:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    # cm = tf.math.confusion_matrix(labels=labels, predictions=pred).numpy()
    # TP = cm[1][1]
    # FP = cm[0][1]
    # FN = cm[1][0]
    # TN = cm[0][0]
    # print(cm)

    # diagonal_sum = cm.trace()
    # sum_of_all_elements = cm.sum()
    
    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    print("=====Prediction result=====")
    print("Confusion Matrix (row1: TN,FP | row2: FN,TP):\n", cm)
    print("AUC: ", roc_auc)
    print("Best Threshold: ", optimal_th)
    print("Accuracy: ", (TP + TN)/DATA_NUM)
    print("Recall (TPR): ", TP/(TP+FN))
    print("TNR: ", TN/(FP+TN))
    print("PPV: ", TP/(TP+FP))
    print("NPV: ", TN/(FN+TN))
    print("False Alarm Rate (FPR): ", FP/(FP+TN))
    print("Leakage Rate (FNR): ", FN/(FN+TP))
    print("F1-Score: ", f1_score(labels, pred_labels)) # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)

if __name__ == "__main__":
    n_dist_list = list()
    s_dist_list = list()
    measure_dist = "normal" # normal & perceptual
    class_name = ["normal", "smura"]
    for mode in class_name:
        #fn_original = glob.glob(f"./test_data/{mode}/*real_B.png")
        #fn_inpaint = glob.glob(f"./test_data/{mode}/*fake_B.png")
        fn_original = glob.glob(f"./results/{mode}/exp/test_latest/images/*real_B.png")
        fn_inpaint = glob.glob(f"./results/{mode}/exp/test_latest/images/*fake_B.png")
        fn_original = sorted(fn_original)
        fn_inpaint = sorted(fn_inpaint)
        print(len(fn_original))
        print(len(fn_inpaint))
        if measure_dist == "perceptual":
            for i in range(len(fn_original)):
                if i == 1:
                    print(fn_original[i])
                    print(fn_inpaint[i])
                img_origin = lpips.im2tensor(lpips.load_image(fn_original[i]))
                img_inpaint = lpips.im2tensor(lpips.load_image(fn_inpaint[i]))

                if(use_gpu):
                    img_origin = img_origin.cuda()
                    img_inpaint = img_inpaint.cuda()

                dist = loss_fn.forward(img_origin, img_inpaint)
                
                if mode == "normal":
                    n_dist_list.append(dist[0,0,0,0].cpu().detach().numpy())
                else:
                    s_dist_list.append(dist[0,0,0,0].cpu().detach().numpy())
        else:
            for i in range(len(fn_original)):
                if i == 1:
                    print(fn_original[i])
                    print(fn_inpaint[i])
                # RGB
                # img_origin = cv2.imread(fn_original[i]).ravel()
                # img_inpaint = cv2.imread(fn_inpaint[i]).ravel()
                # gray
                img_origin = cv2.imread(fn_original[i],0).ravel()
                img_inpaint = cv2.imread(fn_inpaint[i],0).ravel()

                dist = np.linalg.norm(img_inpaint-img_origin)
                
                if mode == "normal":
                    n_dist_list.append(dist)
                else:
                    s_dist_list.append(dist)

    print(len(n_dist_list))
    print(len(s_dist_list))
    n_dist_list = np.array(n_dist_list)
    print(f"mean: {n_dist_list.mean()}, std: {n_dist_list.std()}")
    
    s_dist_list = np.array(s_dist_list)
    print(f"mean: {s_dist_list.mean()}, std: {s_dist_list.std()}")

    true_labels = [0]*n_dist_list.shape[0]+[1]*s_dist_list.shape[0]
    dist_list = np.concatenate([n_dist_list,s_dist_list])
    prediction(true_labels,dist_list,"image_inpainting")