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
    plt.show()
    plt.clf()
def plot_distance_distribution(n_scores, s_scores, name):
    # bins = np.linspace(0.00001,0.0001) # MSE
    # bins = np.linspace(0.00001,0.0001) # D score
    bins = np.linspace(0.00001,0.0001) # 4k
    # bins = np.linspace(0.00001,0.001) # 8k
    plt.hist(s_scores, bins, alpha=0.5, label="smura")
    plt.hist(n_scores, bins, alpha=0.5, label="normal")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Number')
    plt.title('Distribution')
    plt.legend(loc='upper right')
    plt.savefig(name + '_score.png')
    plt.show()
    plt.clf()
def plot_distance_scatter(n_max, s_max, n_mean, s_mean, name):
    # normal
    x1 = n_max
    y1 = n_mean
    # smura
    x2 = s_max
    y2 = s_mean
    # 設定座標軸
    # normal
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.legend(loc='upper right')
    plt.savefig(name + '_normal_scatter.png')
    plt.clf()
    # smura
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(name + '_smura_scatter.png')
    plt.clf()
    # all
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(name + '_scatter.png')
    plt.clf()
def prediction(labels, scores, name):
    pred_labels = [] 
    roc_auc, optimal_th = roc(labels, scores, name)
    for score in scores:
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
def combined_prediction(labels, max_scores, mean_scores, name):
    # score = a*max + b*mean + c
    best_a, best_b, best_c = 0, 0, 0
    best_auc = 0
    for ten_a in range(0, 10, 1):
        a = ten_a/10.0
        for ten_b in range(0, 10, 1):
            b = ten_b/10.0
            for ten_thousand_c in range(-10, 5, 1): # y -0.0001~0.00005
                c = ten_thousand_c/100000
                scores = a*max_scores + b*mean_scores + c
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

    best_scores = best_a*max_scores + best_b*mean_scores + best_c
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
    opt.serial_batches = True  # no shuffle
    # opt.serial_batches = False
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    
    n_mse_list = []
    s_mse_list = []
    for i, data in enumerate(dataset):
        print(f"Image num: {i}")
        s_data = data['smura']
        n_data = data['normal']
        # print(n_data['A'].shape)
        # print(s_data['A'].shape)
    
        # 建立 input real_A & real_B
        # it not only sets the input data with mask, but also sets the latent mask.
        model.set_input(s_data)
        s_mse_list.append(model.test())

        model.set_input(n_data) 
        n_mse_list.append(model.test())

    print(f"smura mean: {np.array(s_mse_list).mean()}")
    print(f"smura std: {np.array(s_mse_list).std()}")
    print(f"normal mean: {np.array(n_mse_list).mean()}")
    print(f"normal std: {np.array(n_mse_list).std()}")

    print(len(s_mse_list))
    print(len(n_mse_list))

    bins = np.linspace(0.00001,0.0001) # 4k
    plt.hist(s_mse_list, bins, alpha=0.5, label="smura")
    plt.hist(n_mse_list, bins, alpha=0.5, label="normal")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Number')
    plt.title('Distribution')
    plt.legend(loc='upper right')
    plt.savefig(f'd12_{opt.measure_mode}_type_c.png')
    plt.show()
    plt.clf()
    
    all_mse = np.concatenate([n_mse_list, s_mse_list])
    true_label = [0]*len(n_mse_list)+[1]*len(s_mse_list)

    prediction(true_label, all_mse, f"d12_{opt.measure_mode}_result")
    
        

       