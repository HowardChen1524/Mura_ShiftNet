#!/usr/bin/env python
# coding: utf-8

# In[78]:


import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix

def roc(labels, scores, name_model):
    
    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]
    
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_th

def plot_distance_distribution(n_mse_log, s_mse_log, name_dist, max_val):
    bins = np.linspace(0, 10)
    # bins = np.linspace(0, max_val)
    plt.hist(s_mse_log, bins, alpha=0.5, label="smura")
    plt.hist(n_mse_log, bins, alpha=0.5, label="normal")
    plt.xlabel('MSE')
    plt.ylabel('Number')
    plt.title('Distance distribution')
    plt.legend(loc='upper right')
    plt.savefig(name_dist + '_dist.png')
    plt.show()
    plt.clf()

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

def prediction(labels, scores, name_model, normal_max):
    pred_labels = [] 
    print(normal_max)
    roc_auc, optimal_th = roc(labels, scores, name_model)
    for score in scores:
        #if score >= optimal_th:
        if score >= normal_max:
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
    print("TPR-FPR Threshold: ", optimal_th)
    print("N_Max Threshold: ", normal_max)
    print("Accuracy: ", (TP + TN)/DATA_NUM)
    print("Recall (TPR): ", TP/(TP+FN))
    print("TNR: ", TN/(FP+TN))
    print("PPV: ", TP/(TP+FP))
    print("NPV: ", TN/(FN+TN))
    print("False Alarm Rate (FPR): ", FP/(FP+TN))
    print("Leakage Rate (FNR): ", FN/(FN+TP))
    print("F1-Score: ", f1_score(labels, pred_labels)) # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)

# In[ ]:


n_mse_log = list()
s_mse_log = list()

# measure_dist = "normal" # rgb & gray & perceptual

# =====0407=====
#class_name = ["normal_ori", "smura_ori"] # 3095
#class_name = ["normal_ori_2", "smura_ori_2"] # 3095
#class_name = ["normal_ori_3", "smura_ori_3"] # 3095
#class_name = ["normal_ori_4", "smura_ori_4"] # 3095
#class_name = ["normal_ori_5", "smura_ori_5"] # 3095

#class_name = ["normal_ori_test_305", "smura_ori_test_305"] # 305
#class_name = ["normal_ori_test_305_2", "smura_ori_test_305_2"] # 305
#class_name = ["normal_ori_test_305_3", "smura_ori_test_305_3"] # 305

# =====0429=====
#class_name = ["normal_ori_345_0429_epoch_60", "smura_ori_345_0429_epoch_60"] # 345
#class_name = ["normal_ori_345_0429_epoch_60_2", "smura_ori_345_0429_epoch_60_2"] # 345
#class_name = ["normal_ori_345_0429_epoch_60_3", "smura_ori_345_0429_epoch_60_3"] # 345

# =====train d12 test d15=====
#class_name = ["normal_ori_train_d12_test_d15", "smura_ori_train_d12_test_d15"] # 345
#class_name = ["normal_ori_train_d12_test_d15_2", "smura_ori_train_d12_test_d15_2"] # 345
#class_name = ["normal_ori_train_d12_test_d15_3", "smura_ori_train_d12_test_d15_3"] # 345
#class_name = ["normal_ori_train_d12_test_d15_4", "smura_ori_train_d12_test_d15_4"] # 345
#class_name = ["normal_ori_train_d12_test_d15_5", "smura_ori_train_d12_test_d15_5"] # 345

# =====train d12 test d17=====
#class_name = ["normal_ori_train_d12_test_d17", "smura_ori_train_d12_test_d17"] # 345
class_name = ["normal_ori_train_d12_test_d17_2", "smura_ori_train_d12_test_d17_2"] # 345

for mode in class_name:
    
    fn_mask = sorted(glob.glob(f"./results/{mode}/exp/test_latest/images/*real_A.png"))
    fn_original = sorted(glob.glob(f"./results/{mode}/exp/test_latest/images/*real_B.png"))
    fn_inpaint = sorted(glob.glob(f"./results/{mode}/exp/test_latest/images/*fake_B.png"))
    if fn_original[0][:-11]==fn_inpaint[0][:-11]:
        print("same file")
    # find mask position
    for i in range(len(fn_original)):
        
        mask = Image.open(fn_mask[i])
        mask = np.array(mask)
        mask = mask[:,:,3]

        pos = list()
        for k, row in enumerate(mask):
            for l, col in enumerate(row):
                if col == 127:
                    pos.append((k,l))
        start_row = pos[0][0]
        end_row = pos[-1][0]
        start_col = pos[0][1]
        end_col = pos[-1][1]
        #print(len(pos))
        del pos
        
        # crop mask image
        real = Image.open(fn_original[i])
        real = np.array(real).astype("float32")
        real_mask = real[start_row:end_row, start_col:end_col]
        
        fake = Image.open(fn_inpaint[i])
        fake = np.array(fake).astype("float32")
        fake_mask = fake[start_row:end_row, start_col:end_col]
        del real, fake
        
        # store mask mean std
#         print(f"smura real_mask mean: {real_mask.mean()}")
#         print(f"smura real_mask std: {real_mask.std()}")
#         print(f"smura fake_mask mean: {fake_mask.mean()}")
#         print(f"smura fake_mask std: {fake_mask.std()}")
        
        # compute mask dist
        mse = MeanSquaredError()
        if mode == class_name[0]:
            mse_loss = mse(real_mask,fake_mask)
            n_mse_log.append(mse_loss)
        elif mode == class_name[1]:
            mse_loss = mse(real_mask,fake_mask)
            s_mse_log.append(mse_loss)
        '''
        #Not Mask
        real = Image.open(fn_original[i])
        real = np.array(real).astype("float32")
        
        fake = Image.open(fn_inpaint[i])
        fake = np.array(fake).astype("float32")

        # compute mask dist
        mse = MeanSquaredError()
        if mode == class_name[0]:
            mse_loss = mse(real,fake)
            n_mse_log.append(mse_loss)
        elif mode == class_name[1]:
            mse_loss = mse(real,fake)
            s_mse_log.append(mse_loss)
        '''

n_mse_log_arr = np.array(n_mse_log)
s_mse_log_arr = np.array(s_mse_log)

dists = np.concatenate([n_mse_log_arr, s_mse_log_arr])
true_label = [0]*n_mse_log_arr.shape[0]+[1]*s_mse_log_arr.shape[0]

plot_distance_distribution(n_mse_log, s_mse_log, "MSE", np.max(dists))

prediction(true_label, dists, "mask_distance", np.max(n_mse_log_arr))


# In[76]:
