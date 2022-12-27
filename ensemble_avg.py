#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics
from tqdm import tqdm
import torch

# ## 載入圖片預測結果
data1_df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k/model_0/sup_conf.csv')
data2_df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k/model_1/sup_conf.csv')
data3_df = pd.read_csv('/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k/model_2/sup_conf.csv')

save_path = '/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/ensemble_d23/d24_d25_8k'

# ## 預測結果平均
combine_df = data1_df.copy()
combine_df = combine_df.drop(columns=['conf'])
combine_df['ComnbineConf'] = (data1_df['conf']+data2_df['conf']+data3_df['conf'])/3
print(combine_df)


# ## 直接執行
def plot_roc_curve(labels_res, preds_res):
    fpr, tpr, threshold = metrics.roc_curve(y_true=labels_res, y_score=preds_res)
    roc_auc = metrics.auc(fpr, tpr)
    # report = pd.DataFrame(zip(list(tpr), list(fpr), list(threshold)),columns={'TPR','FPR','THRESHOLD'} )
    # report.to_csv("./roc_report_b2429_train.csv", index=False)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    return plt

def calc_metric(labels_res, pred_res, threshold):
    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels_res, y_pred=(pred_res >= threshold)).ravel()
    tpr = tp / (tp + fn) if tp != 0 else 0
    tnr = tn / (tn + fp) if tn != 0 else 0
    precision = tp / (tp + fp) if tp != 0 else 0
    recall = tp / (tp + fn) if tp != 0 else 0
    return threshold, tpr, tnr, precision, recall

def get_curve_df(labels_res, preds_res):
    pr_list = []

    for i in tqdm(np.linspace(0, 1, num=10001)):
        pr_result = calc_metric(labels_res, preds_res, i)
        pr_list.append(pr_result)

    curve_df = pd.DataFrame(pr_list, columns=['threshold', 'tpr', 'tnr', 'precision', 'recall'])
    
    return curve_df

def calc_matrix(labels_res, preds_res):
    results = {'accuracy': [],
           'balance_accuracy': [],
           'tpr': [],
           'tnr': [],
           'tnr0.99_precision': [],
           'tnr0.99_recall': [],
           'tnr0.995_precision': [],
           'tnr0.995_recall': [],
           'tnr0.999_precision': [],
           'tnr0.999_recall': [],
           'precision': [],
           'recall': []
    }

    tn, fp, fn, tp = metrics.confusion_matrix(y_true=labels_res, y_pred=(preds_res >= 0.5)).ravel()
    tpr = tp / (tp + fn) if tp != 0 else 0
    tnr = tn / (fp + tn) if tn != 0 else 0
    fnr = fn / (tp + fn) if fn != 0 else 0
    fpr = fp / (fp + tn) if fp != 0 else 0

    results['accuracy'].append((tn + tp) / (tn + fp + fn + tp))
    results['tpr'].append(tpr)
    results['tnr'].append(tnr) 
    results['balance_accuracy'].append(((tp / (tp + fn) + tn / (tn + fp)) / 2))
    results['precision'].append(tp / (tp + fp))
    results['recall'].append(tp / (tp + fn))

    curve_df = get_curve_df(labels_res, preds_res)
    results['tnr0.99_recall'].append((((curve_df[curve_df['tnr'] > 0.99].iloc[0]) + (curve_df[curve_df['tnr'] < 0.99].iloc[-1])) / 2).recall)
    results['tnr0.995_recall'].append((((curve_df[curve_df['tnr'] > 0.995].iloc[0]) + (curve_df[curve_df['tnr'] < 0.995].iloc[-1])) / 2).recall)
    results['tnr0.999_recall'].append((((curve_df[curve_df['tnr'] > 0.999].iloc[0]) + (curve_df[curve_df['tnr'] < 0.999].iloc[-1])) / 2).recall)
    results['tnr0.99_precision'].append((((curve_df[curve_df['tnr'] > 0.99].iloc[0]) + (curve_df[curve_df['tnr'] < 0.99].iloc[-1])) / 2).precision)
    results['tnr0.995_precision'].append((((curve_df[curve_df['tnr'] > 0.995].iloc[0]) + (curve_df[curve_df['tnr'] < 0.995].iloc[-1])) / 2).precision)
    results['tnr0.999_precision'].append((((curve_df[curve_df['tnr'] > 0.999].iloc[0]) + (curve_df[curve_df['tnr'] < 0.999].iloc[-1])) / 2).precision)
    
    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)

    model_report = pd.DataFrame(results).T
    
    return model_report, curve_df

def get_pred_result(model, dataloader):
    preds_res = []
    labels_res = []
    names_res = []

    with torch.no_grad():
        for i, (inputs, labels, names) in tqdm(enumerate(dataloader)):
            inputs = inputs.to(torch.float32).cuda()
            labels = labels.to(torch.float32).cuda()

            preds = model(inputs)

            preds = torch.reshape(preds, (-1,)).cpu()
            labels = labels.cpu()

            preds_res.extend(preds)
            labels_res.extend(labels)
            names_res.extend(list(names))

    preds_res = np.array(preds_res)
    labels_res = np.array(labels_res)

    return labels_res, preds_res, names_res

def predict_report(preds, labels, names):
    list_combined = []
    list_combined = list(zip(names, preds, labels))
    df_res = pd.DataFrame(list_combined, columns=['name', 'conf', 'label'])
    return df_res

# ## save_path 記得改
files_res = np.array(combine_df.loc[:, 'name'])
labels_res = np.array(combine_df.loc[:, 'label'])
pred = np.array(combine_df.loc[:, 'ComnbineConf'])
prefix = 'test'


os.makedirs(save_path, exist_ok=True)
preds_res = pred
        
model_pred_result = predict_report(preds_res, labels_res, files_res)
model_pred_result.to_csv(os.path.join(save_path, f'sup_conf.csv'), index=None)
print("model predict record finished!")

fig = plot_roc_curve(labels_res, preds_res)
fig.savefig(os.path.join(save_path, f'{prefix}_roc_curve.png'))
print("roc curve saved!")

# In[ ]:




