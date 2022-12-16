import os
import random
from collections import defaultdict
from tqdm import tqdm
import time

import torch
import numpy as np
import pandas as pd

# from core.dataset import AUO_Dataset, AI9_Dataset
from data.AI9_dataset import AI9_Dataset

import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import DataLoader
import torchvision
import tensorflow as tf
import tensorflow_addons as tfa
from PIL import Image, ImageEnhance

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

# ***** convenient func *********************************************
def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def tensor2img(input_image):
    if isinstance(input_image, torch.Tensor): 
        image_tensor = input_image.detach().cpu()
    else:
        raise
    transform = transforms.Compose([transforms.ToPILImage()])
    image = transform((image_tensor+1) / 2.0)
    return image

def enhance_img(img,factor=5):
  enh_con = ImageEnhance.Contrast(img)
  new_img = enh_con.enhance(factor=factor)
  return new_img

def set_seed(seed, base=0, is_set=True):
  seed += base # 2022 + 0
  assert seed >=0, '{} >= {}'.format(seed, 0)
  # 在神经网络中，参数默认是进行随机初始化的。如果不设置的话每次训练时的初始化都是随机的，导致结果不确定。如果设置初始化，则每次初始化都是固定的。
  torch.manual_seed(seed) # #为CPU设置种子用于生成随机数，以使得结果是确定的
  torch.cuda.manual_seed_all(seed) #为所有GPU设置随机种子；torch.cuda.manual_seed() 为當前GPU设置随机种子
  random.seed(seed) # seed固定的話，生成同一个随机数
  # torch.backends.cudnn.deterministic是啥？顾名思义，将这个 flag 置为True的话，每次返回的卷积算法将是确定的，即默认算法。cudnn中包含很多卷积算法。基于 GEMM (General Matrix Multiply) 的，基于 FFT 的，基于 Winograd 算法的等等。
  '''
  cudnn中卷积算法
  static const algo_t algos[] = {
         CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT,
         CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,#默认
         CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
         CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
    };
  '''
  torch.backends.cudnn.deterministic = True
  # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。适用场景是网络结构固定（不是动态变化的），网络的输入形状（包括 batch size，图片大小，输入的通道）是不变的，其实也就是一般情况下都比较适用。反之，如果卷积层的设置一直变化，将会导致程序不停地做优化，反而会耗费更多的时间。
  torch.backends.cudnn.benchmark = True
# ***** origin supervised func **************************************************

class wei_augumentation(object):
    def __call__(self, img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)
        img2 = tf.image.sobel_edges(img[None, ...])
        equal_img = tfa.image.equalize(img, bins=256)
        img = tf.concat([equal_img, img2[0, :, :, 0]], 2)
        image_array  = tf.keras.preprocessing.image.array_to_img(img)
        
        return image_array
    def __repr__(self):
        return self.__class__.__name__+'()'

class tjwei_augumentation(object):
    def __call__(self, img):
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)
        img2 = tf.concat([img, img], 2)
        img = tf.concat([img, img2], 2)
        # img2 = tf.image.sobel_edges(img[None, ...])
        # img = tf.concat([img, img2[0, :, :, 0]], 2)
        image_array = tf.keras.preprocessing.image.array_to_img(img)
        
        return image_array
    def __repr__(self):
        return self.__class__.__name__+'()'

data_transforms = {
    "train": transforms.Compose([
        # transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # tjwei_augumentation(),
        # transforms.ToTensor(),
    ]),
    "test": transforms.Compose([
        transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
        tjwei_augumentation(),
        transforms.ToTensor()
    ])
}

def make_training_dataloader(ds):
    mura_ds = ds["train"]["mura"]
    normal_ds = ds["train"]["normal"]
    min_len = min(len(mura_ds), len(normal_ds))
    sample_num = int(4 * min_len)
    # sample_num = 32
    normal_ds = torch.utils.data.Subset(normal_ds,random.sample(list(range(len(normal_ds))), sample_num))
    train_ds = torch.utils.data.ConcatDataset([mura_ds, normal_ds])
    # train_ds = torch.utils.data.ConcatDataset([normal_ds])
    dataloader = DataLoader(train_ds, 
                            batch_size=16,
                            shuffle=True, 
                            num_workers=0,
                           )
    return dataloader

def make_test_dataloader(ds):
    m = ds["test"]["mura"]
    n = ds["test"]["normal"]
    s_dataloader = DataLoader(m, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=0,
                           )
    n_dataloader = DataLoader(n, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=0,
                           )
    return [n_dataloader, s_dataloader]

def make_val_dataloader(ds):
    m = ds["val"]["mura"]
    n = ds["val"]["normal"]
    val_ds = torch.utils.data.ConcatDataset([m, n])
    dataloader = DataLoader(val_ds, 
                            batch_size=4,
                            shuffle=False, 
                            num_workers=0,
                           )
    return dataloader

def predict_report(preds, labels, names):
    df_res = pd.DataFrame(list(zip(names, preds)), columns=["Img", "Predict"])
    df_res["Label"] = labels
    return df_res

def get_curve_df(labels_res, preds_res):
    pr_list = []

    for i in tqdm(np.linspace(0, 1, num=10001)):
        pr_result = calc_metric(labels_res, preds_res, i)
        pr_list.append(pr_result)

    curve_df = pd.DataFrame(pr_list, columns=['threshold', 'tnr', 'precision', 'recall', 'f1', 'fpr'])
    
    return curve_df

def calc_metric(labels_res, pred_res, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true=labels_res, y_pred=(pred_res >= threshold)).ravel()
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    return threshold, tnr, precision, recall, f1, fpr

def calc_matrix(labels_res, preds_res):
    results = {
                'tnr0.987_th': [],
                'tnr0.987_tnr': [],
                'tnr0.987_precision': [],
                'tnr0.987_recall': [],
                'tnr0.987_f1': [],
                'tnr0.987_fpr': [],
                'tnr0.996_th': [],
                'tnr0.996_tnr': [],
                'tnr0.996_precision': [],
                'tnr0.996_recall': [],
                'tnr0.996_f1': [],
                'tnr0.996_fpr': [],
                'tnr0.998_th': [],
                'tnr0.998_tnr': [],
                'tnr0.998_precision': [],
                'tnr0.998_recall': [],
                'tnr0.998_f1': [],
                'tnr0.998_fpr': [],
              }

    curve_df = get_curve_df(labels_res, preds_res)
    
    results['tnr0.987_th'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).threshold)
    results['tnr0.987_tnr'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).tnr)
    results['tnr0.987_recall'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).recall)
    results['tnr0.987_precision'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).precision)
    results['tnr0.987_f1'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).f1)
    results['tnr0.987_fpr'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).fpr)

    results['tnr0.996_th'].append((curve_df[curve_df['tnr'] > 0.996].iloc[0]).threshold)
    results['tnr0.996_tnr'].append((curve_df[curve_df['tnr'] > 0.996].iloc[0]).tnr)
    results['tnr0.996_recall'].append((curve_df[curve_df['tnr'] > 0.996].iloc[0]).recall)
    results['tnr0.996_precision'].append((curve_df[curve_df['tnr'] > 0.996].iloc[0]).precision)
    results['tnr0.996_f1'].append((curve_df[curve_df['tnr'] > 0.996].iloc[0]).f1)
    results['tnr0.996_fpr'].append((curve_df[curve_df['tnr'] > 0.996].iloc[0]).fpr)

    results['tnr0.998_th'].append((curve_df[curve_df['tnr'] > 0.998].iloc[0]).threshold)
    results['tnr0.998_tnr'].append((curve_df[curve_df['tnr'] > 0.998].iloc[0]).tnr)
    results['tnr0.998_recall'].append((curve_df[curve_df['tnr'] > 0.998].iloc[0]).recall)
    results['tnr0.998_precision'].append((curve_df[curve_df['tnr'] > 0.998].iloc[0]).precision)
    results['tnr0.998_f1'].append((curve_df[curve_df['tnr'] > 0.998].iloc[0]).f1)
    results['tnr0.998_fpr'].append((curve_df[curve_df['tnr'] > 0.998].iloc[0]).fpr)

    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)

    model_report = pd.DataFrame(results)
    
    return results, model_report, curve_df
    
def get_data_info(t, l, image_info, data_dir, csv_path):
    res = []
    # image_info = image_info[(image_info["train_type"] == t) & (image_info["label"] == l) & (image_info["PRODUCT_CODE"] == "T850MVR05")]
    image_info = image_info[(image_info["batch"] >= 24) & (image_info["batch"] <= 25) & (image_info["label"] == l) & (image_info["PRODUCT_CODE"] == "T850MVR05")]
    # print(image_info)
    
    for path, img, label, JND in zip(image_info["path"],image_info["name"],image_info["label"],image_info["MULTI_JND"]):
        img_path = os.path.join(os.path.join(data_dir), path, img)
        res.append([img_path, label, JND, t, img])
    X = []
    Y = []
    N = []
    
    for d in res:
        # dereference ImageFile obj
        X.append(os.path.join(d[0]))
        Y.append(d[1])
        N.append(d[4])

    dataset = AI9_Dataset(feature=X,
                          target=Y,
                          name=N,
                          transform=data_transforms[t])
    # print(dataset.__len__())

    return dataset

def evaluate(model, testloaders, save_path):
    model.eval().cuda()
    res = defaultdict(dict)
    for l in ['preds_res','labels_res','files_res']:
      for t in ['n', 's']:
        res[l][t] = []

    with torch.no_grad():
      for idx, loader in enumerate(testloaders):
        for inputs, labels, names in tqdm(loader):
            
          inputs = inputs.cuda()
          labels = labels.cuda()
          
          preds = model(inputs)
          
          preds = torch.reshape(preds, (-1,)).cpu()
          labels = labels.cpu()
          
          names = list(names)

          if idx == 0:
            res['files_res']['n'].extend(names)
            res['preds_res']['n'].extend(preds)
            res['labels_res']['n'].extend(labels)
          elif idx == 1:
            res['files_res']['s'].extend(names)
            res['preds_res']['s'].extend(preds)
            res['labels_res']['s'].extend(labels)
          
    res['files_res']['all'] = res['files_res']['n'] + res['files_res']['s'] # list type
    res['preds_res']['all'] = np.array(res['preds_res']['n'] + res['preds_res']['s'])
    res['labels_res']['all'] = np.array(res['labels_res']['n'] + res['labels_res']['s'])
    
    calc_roc(res['labels_res']['all'], res['preds_res']['all'], save_path, "sup")
    print("roc curve saved!")

    return res

def find_sup_th(res, save_path):
    # model_pred_result = predict_report(res['preds_res']['all'], res['labels_res']['all'], res['files_res']['all'])
    # model_pred_result.to_csv(os.path.join(save_path, "sup_model_pred_result.csv"), index=None)
    # print("model predict record finished!")
    all_label = res['label']['n'] + res['label']['s']
    all_conf = res['conf']['n'] + res['conf']['s']
    res, model_report, curve_df = calc_matrix(all_label, all_conf)
    model_report.to_csv(os.path.join(save_path, "sup_model_report.csv"))
    curve_df.to_csv(os.path.join(save_path, "sup_model_precision_recall_curve.csv"))
    print("model report record finished!")
    
    return res
# *****************************************************************


# dataset
def get_data_info_unsup(t, image_info, csv_path):
    pass
    # res = []
    # image_info = image_info[(image_info["train_type"] == t) & (image_info["batch"] <= 23) & (image_info["PRODUCT_CODE"] == "T850MVR05")]
    # # image_info = image_info[(image_info["batch"] >= 24) & (image_info["batch"] <= 25) & (image_info["label"] == l) & (image_info["PRODUCT_CODE"] == "T850MVR05")]

    # for path, img, label, JND, t in zip(image_info["path"],image_info["name"],image_info["label"],image_info["MULTI_JND"],image_info["train_type"]):
    #     img_path = os.path.join(os.path.dirname(csv_path), path, img)
    #     res.append([img_path, label, JND, t, img])
    # X = []
    # Y = []
    # N = []
    
    # for d in res:
    #     # dereference ImageFile obj
    #     X.append(os.path.join(d[0]))
    #     Y.append(d[1])
    #     N.append(d[4])
    # dataset = AUO_Dataset(feature=X,
    #                       target=Y,
    #                       name=N,
    #                       transform=data_transforms[t])
    # # print(dataset.__len__())
    # return dataset

# calculate metric
def calc_roc(labels, scores, path, name):
    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    plot_roc_curve(roc_auc, fpr, tpr, path, name)

    return roc_auc, optimal_th

def unsup_calc_metric(labels, scores, path, name):
    result_msg = ''
    pred_labels = [] 
    roc_auc, optimal_th = calc_roc(labels, scores, path, name)
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
    result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
    result_msg += f"\nAUC: {roc_auc}\n"
    result_msg += f"Threshold (highest TPR-FPR): {optimal_th}\n"
    result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
    result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
    result_msg += f"TNR: {TN/(FP+TN)}\n"
    result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
    result_msg += f"NPV: {TN/(FN+TN)}\n"
    result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
    result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
    result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
    
    return result_msg

def unsup_find_param_max_mean(labels, max_scores, mean_scores, path, name):
    result_msg = ''
    best_a = best_b = 0
    best_auc = 0
    for ten_a in range(0, 11, 1):
        a = ten_a/10
        for ten_b in range(0, 11, 1):
            b = ten_b/10
            scores = a*max_scores + b*mean_scores
            fpr, tpr, _ = roc_curve(labels, scores)
            current_auc = auc(fpr, tpr)
            if current_auc >= best_auc:
                best_auc = current_auc
                best_a = a
                best_b = b         

    best_scores = best_a*max_scores + best_b*mean_scores

    result_msg += f"Param a: {best_a}, b: {best_b}\n"
    result_msg += unsup_calc_metric(labels, best_scores, path, name)

    return result_msg

# find threshold or using threshold to prediction
def sup_unsup_prediction_spec_th(labels, all_conf_sup, all_score_unsup, threshold, path):
    result_msg = ''

    th_list = [threshold['tnr0.987'], threshold['tnr0.996'], threshold['tnr0.998']]  
    for th in th_list:
        pred_labels = [] 
        combined_scores = th['m']*all_score_unsup + all_conf_sup
        for score in combined_scores:
            if score >= th['b']:
                pred_labels.append(1)
            else:
                pred_labels.append(0)

        cm = confusion_matrix(labels, pred_labels)
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[0][0]
        DATA_NUM = TN + FP + FN + TP
        
        result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
        result_msg += f"\nThreshold line: m:{th['m']}, b:{th['b']}\n"
        result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
        result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
        result_msg += f"TNR: {TN/(FP+TN)}\n"
        result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
        result_msg += f"NPV: {TN/(FN+TN)}\n"
        result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
        result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
        result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
        result_msg += f"===================================\n"

    return result_msg

def sup_unsup_prediction_spec_multi_th(labels, all_conf_sup, all_score_unsup, threshold, path):
    result_msg = ''

    th_list = [threshold['tnr0.987'], threshold['tnr0.996'], threshold['tnr0.998']] 
    for th in th_list:
        pred_labels = [] 

        combined_scores_1 = th['m']*all_score_unsup + all_conf_sup
        combined_scores_2 = all_score_unsup
        
        for score_1, score_2 in zip(combined_scores_1, combined_scores_2):
            if score_1 < th['b'] and score_2 < th['x']:
                pred_labels.append(0)
            else:
                pred_labels.append(1)
        
        cm = confusion_matrix(labels, pred_labels)
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[0][0]
        DATA_NUM = TN + FP + FN + TP
        
        result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
        result_msg += f"\nThreshold line: m:{th['m']}, b:{th['b']}\n"
        result_msg += f"Threshold line: {th['x']}\n"
        result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
        result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
        result_msg += f"TNR: {TN/(FP+TN)}\n"
        result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
        result_msg += f"NPV: {TN/(FN+TN)}\n"
        result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
        result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
        result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
        result_msg += f"===================================\n"

    return result_msg

def sup_prediction_spec_th(labels, all_conf_sup, threshold, path):
    result_msg = ''

    th_list = [threshold['tnr0.987'], threshold['tnr0.996'], threshold['tnr0.998']]  
    for th in th_list:
        cm = confusion_matrix(labels, (all_conf_sup >= th))
        #[[TN,FP]
        # [FN,TP]]
        TP = cm[1][1]
        FP = cm[0][1]
        FN = cm[1][0]
        TN = cm[0][0]
        DATA_NUM = TN + FP + FN + TP
        
        result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
        result_msg += f"\nThreshold : {th}\n"
        result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
        result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
        result_msg += f"TNR: {TN/(FP+TN)}\n"
        result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
        result_msg += f"NPV: {TN/(FN+TN)}\n"
        result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
        result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
        result_msg += f"F1-Score: {f1_score(labels, (all_conf_sup >= th))}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
        result_msg += f"===================================\n"

    return result_msg

def sup_unsup_prediction_spec_th_manual(labels, all_conf_sup, all_score_unsup, th, path):
    result_msg = ''
    pred_labels = [] 
    combined_scores = th['m']*all_score_unsup + all_conf_sup
    for score in combined_scores:
        if score >= th['b']:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    
    result_msg += f"Confusion Matrix (row1: TN,FP | row2: FN,TP):\n{cm}"
    result_msg += f"\nThreshold line: m:{th['m']}, b:{th['b']}\n"
    result_msg += f"Accuracy: {(TP + TN)/DATA_NUM}\n"
    result_msg += f"Recall (TPR): {TP/(TP+FN)}\n"
    result_msg += f"TNR: {TN/(FP+TN)}\n"
    result_msg += f"Precision (PPV): {TP/(TP+FP)}\n"
    result_msg += f"NPV: {TN/(FN+TN)}\n"
    result_msg += f"False Alarm Rate (FPR): {FP/(FP+TN)}\n"
    result_msg += f"Leakage Rate (FNR): {FN/(FN+TP)}\n"
    result_msg += f"F1-Score: {f1_score(labels, pred_labels)}\n" # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
    result_msg += f"===================================\n"

    return result_msg

def sup_unsup_prediction_auto_th(labels, all_conf_sup, all_score_unsup, path):
    all_pr_res = []
    # mx + y = b
    # score 同乘 100000 4e-05 ~ 4e-04
    # m 0~-50000 stride = 0.01 
    # b 0~20 stride = 0.1
    start_time = time.time()
    for times_m in range(0, 501, 1): # 次數
    # for times_m in range(0, 251, 1): # resunet
    # for times_m in range(0, 1000, 1): # skipgan
        m = (100)*times_m
        # m = (10)*times_m # resunet
        # m = (100)*times_m # skipgan
        for times_b in range(0, 2001, 1): # 次數      
            b = (0.01)*times_b
            combined_scores = m*all_score_unsup + all_conf_sup
            tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=(combined_scores >= b)).ravel()
            tnr = tn / (tn + fp)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)
            fpr = fp / (fp + tn)
            print(tnr)
            if tnr >= 0.987:
                all_pr_res.append([m, b, tnr, precision, recall, f1, fpr])
    total_time = time.time() - start_time

    res = save_curve_and_report(all_pr_res, path)
    return res, total_time

def sup_unsup_prediction_auto_multi_th(labels, all_conf_sup, all_score_unsup, path):
    all_pr_res = []
    # one 垂直 one 斜線
    # mx + y = b
    # score 同乘 100000 3e-05 ~ 1.2e-04 -> 3~12
    # m 0~-1 stride = 0.01 
    # b 3~12 stride = 0.1
    # x axis score 0.000055~0.00007
    start_time = time.time()
    for times_x in range(0, 11, 1): # 次數
        x = 0.0001 + (times_x*0.000005)
        # x = 0.0006 + (times_x*0.00005) # resunet
        # x = 0.000009 + (times_x*0.0000002) # skipgan
        for times_m in range(0, 51, 1): # 次數
        # for times_m in range(0, 25, 1): # resunet
        # for times_m in range(0, 100, 1): # skipgan
            m = (1000)*times_m
            # m = (100)*times_m
            # m = (1000)*times_m
            for times_b in range(0, 2001, 1): # 次數      
                b = (0.01)*times_b
            
                pred_labels = [] 
                combined_scores = m*all_score_unsup + all_conf_sup
                for score, score_x in zip(combined_scores, all_score_unsup):
                    if score < b and score_x < x:
                        pred_labels.append(0)
                    else:
                        pred_labels.append(1)
                
                tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
                tnr = tn / (tn + fp)
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision * recall) / (precision + recall)
                fpr = fp / (fp + tn)
                print(tnr)
                if tnr >= 0.987:
                    print(tnr)
                    all_pr_res.append([m, b, x, tnr, precision, recall, f1, fpr])
    total_time = time.time() - start_time

    res = save_curve_and_report(all_pr_res, path, False)
    return res, total_time

def sup_unsup_svm(true_label, all_conf_sup, all_score_unsup, path):
    X = list(zip(all_score_unsup, all_conf_sup))
    y = true_label

    svm = LinearSVC(random_state=0)
    svm.fit(X,y)
    # print(svm.coef_)
    # print(svm.intercept_)
    m = -(svm.coef_[0][0]/svm.coef_[0][1])
    b = -(svm.intercept_[0]/svm.coef_[0][1])
    # print(m)
    # print(b)
    # print(f"y = {m}x + {b}")
    # y = -0.9397935634917242x + 0.7620793414417739
    # 0, 0.7620793414417739
    # 0.8109007882649589, 0
    y_pred = svm.predict(X)

    tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
    tnr = tn / (tn + fp)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    
    log_name = os.path.join(path, 'svm_report.txt')
    msg = ''
    with open(log_name, "w") as log_file:
        msg += f"y = {m}x + {b}\n"
        msg += f"TNR: {tnr}\n"
        msg += f"PPV: {precision}\n"
        msg += f"TPR: {recall}\n"
        msg += f"F1: {f1}\n"
        msg += f"FPR: {fpr}\n"
        log_file.write(msg)

def get_line_threshold(path):
    one_line_df = pd.read_csv(os.path.join(path, 'model_report.csv'))
    two_line_df = pd.read_csv(os.path.join(path, 'model_report_multi.csv'))
    
    one_line_th = defaultdict(dict)
    for tnr in ['tnr0.987', 'tnr0.996','tnr0.998']:
        for l in ['m', 'b']:
            one_line_th[tnr][l] = None

    one_line_th['tnr0.987']['m'] = one_line_df['tnr0.987_slope'].values[0]
    one_line_th['tnr0.987']['b'] = one_line_df['tnr0.987_threshold'].values[0]
    one_line_th['tnr0.996']['m'] = one_line_df['tnr0.996_slope'].values[0]
    one_line_th['tnr0.996']['b'] = one_line_df['tnr0.996_threshold'].values[0]
    one_line_th['tnr0.998']['m'] = one_line_df['tnr0.998_slope'].values[0]
    one_line_th['tnr0.998']['b'] = one_line_df['tnr0.998_threshold'].values[0]

    two_line_th = defaultdict(dict)
    for tnr in ['tnr0.987', 'tnr0.996','tnr0.998']:
        for l in ['m','b','x']:
            two_line_th[tnr][l] = None

    two_line_th['tnr0.987']['m'] = two_line_df['tnr0.987_slope'].values[0]
    two_line_th['tnr0.987']['b'] = two_line_df['tnr0.987_threshold'].values[0]
    two_line_th['tnr0.987']['x'] = two_line_df['tnr0.987_x'].values[0]
    two_line_th['tnr0.996']['m'] = two_line_df['tnr0.996_slope'].values[0]
    two_line_th['tnr0.996']['b'] = two_line_df['tnr0.996_threshold'].values[0]
    two_line_th['tnr0.996']['x'] = two_line_df['tnr0.996_x'].values[0]
    two_line_th['tnr0.998']['m'] = two_line_df['tnr0.998_slope'].values[0]
    two_line_th['tnr0.998']['b'] = two_line_df['tnr0.998_threshold'].values[0]
    two_line_th['tnr0.998']['x'] = two_line_df['tnr0.998_x'].values[0]

    return one_line_th, two_line_th

def get_value_threshold(path):
    value_df = pd.read_csv(os.path.join(path, 'sup_model_report.csv'))
    
    value_th = defaultdict(dict)
    for tnr in ['tnr0.987', 'tnr0.996','tnr0.998']:
        value_th[tnr] = None

    value_th['tnr0.987'] = value_df['tnr0.987_th'].values[0]
    value_th['tnr0.996'] = value_df['tnr0.996_th'].values[0]
    value_th['tnr0.998'] = value_df['tnr0.998_th'].values[0]

    return value_th

def save_curve_and_report(all_pr_res, path, isOneline=True):
    if isOneline:
        curve_df = pd.DataFrame(all_pr_res, columns=['slope', 'threshold', 'tnr', 'precision', 'recall', 'f1', 'fpr'])
        curve_df.to_csv(os.path.join(path, "model_curve.csv"), index=False)
        print("model curve record finished!")
        results = {
                'tnr0.987_slope': [],
                'tnr0.987_threshold': [],
                'tnr0.987_tnr': [],
                'tnr0.987_precision': [],
                'tnr0.987_recall': [],
                'tnr0.987_f1': [],
                'tnr0.987_fpr': [],
                'tnr0.996_slope': [],
                'tnr0.996_threshold': [],
                'tnr0.996_tnr': [],
                'tnr0.996_precision': [],
                'tnr0.996_recall': [],
                'tnr0.996_f1': [],
                'tnr0.996_fpr': [],
                'tnr0.998_slope': [],
                'tnr0.998_threshold': [],
                'tnr0.998_tnr': [],
                'tnr0.998_precision': [],
                'tnr0.998_recall': [],
                'tnr0.998_f1': [],
                'tnr0.998_fpr': [],
             }
    else:
        curve_df = pd.DataFrame(all_pr_res, columns=['slope', 'threshold', 'x', 'tnr', 'precision', 'recall', 'f1', 'fpr'])
        curve_df.to_csv(os.path.join(path, "model_curve_multi.csv"), index=False)
        print("model curve record finished!")
        results = {
                'tnr0.987_slope': [],
                'tnr0.987_threshold': [],
                'tnr0.987_x': [],
                'tnr0.987_tnr': [],
                'tnr0.987_precision': [],
                'tnr0.987_recall': [],
                'tnr0.987_f1': [],
                'tnr0.987_fpr': [],
                'tnr0.996_slope': [],
                'tnr0.996_threshold': [],
                'tnr0.996_x': [],
                'tnr0.996_tnr': [],
                'tnr0.996_precision': [],
                'tnr0.996_recall': [],
                'tnr0.996_f1': [],
                'tnr0.996_fpr': [],
                'tnr0.998_slope': [],
                'tnr0.998_threshold': [],
                'tnr0.998_x': [],
                'tnr0.998_tnr': [],
                'tnr0.998_precision': [],
                'tnr0.998_recall': [],
                'tnr0.998_f1': [],
                'tnr0.998_fpr': [],
                }
    
    tnr987_best_recall_pos = curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].recall.argmax()
    tnr996_best_recall_pos = curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].recall.argmax()
    tnr998_best_recall_pos = curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].recall.argmax()
    
    if not isOneline:
        results['tnr0.987_x'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).x)
        results['tnr0.996_x'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).x)
        results['tnr0.998_x'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).x)

    results['tnr0.987_slope'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).slope)
    results['tnr0.987_threshold'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).threshold)
    results['tnr0.987_tnr'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).tnr)
    results['tnr0.987_recall'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).recall)
    results['tnr0.987_precision'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).precision)
    results['tnr0.987_f1'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).f1)
    results['tnr0.987_fpr'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).fpr)

    results['tnr0.996_slope'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).slope)
    results['tnr0.996_threshold'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).threshold)
    results['tnr0.996_tnr'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).tnr)
    results['tnr0.996_recall'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).recall)
    results['tnr0.996_precision'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).precision)
    results['tnr0.996_f1'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).f1)
    results['tnr0.996_fpr'].append((curve_df[(curve_df['tnr'] > 0.996) & (curve_df['tnr'] < 0.997)].iloc[tnr996_best_recall_pos]).fpr)

    results['tnr0.998_slope'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).slope)
    results['tnr0.998_threshold'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).threshold)
    results['tnr0.998_tnr'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).tnr)
    results['tnr0.998_recall'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).recall)
    results['tnr0.998_precision'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).precision)
    results['tnr0.998_f1'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).f1)
    results['tnr0.998_fpr'].append((curve_df[(curve_df['tnr'] > 0.998) & (curve_df['tnr'] < 0.999)].iloc[tnr998_best_recall_pos]).fpr)
    
    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)
    
    model_report = pd.DataFrame(results)
    if isOneline:
        model_report.to_csv(os.path.join(path, "model_report.csv"))
    else:
        model_report.to_csv(os.path.join(path, "model_report_multi.csv"))
    print("model report record finished!")
    return results

# draw image
def plot_one_line(line, color):
    # mx + y = b 
    # x = 1.2e-04, y = b - 1.2e-04m
    # y = 1, x = (b-y)/m
    # y = 0, x = b/m
    slope, intercept = line
    if slope == 0:
        # x_vals = [0, 4e-04]
        x_vals = [0, 1.5e-04]
        y_vals = [intercept, intercept]
    else:
        x_vals = [(intercept-1)/slope, intercept/slope]
        y_vals = [1, 0]
    plt.plot(x_vals, y_vals, color=color)

def plot_two_line(line, color):
    # mx + y = b 
    # x = 1, y = b - m
    # y = 1, x = (b-y)/m
    slope, intercept, x = line
    if slope == 0:
        # x_vals = [0, 4e-04]
        x_vals = [0, 1.5e-04]
        y_vals = [intercept, intercept]
    else:
        x_vals = [(intercept-1)/slope, intercept/slope]
        y_vals = [1, 0]
    plt.plot(x_vals, y_vals, color=color)
    # one vertical line x
    x_vertical = [x, x]
    y_vertical = [1, 0]
    plt.plot(x_vertical, y_vertical, color=color)
    
def plot_scatter(conf_sup, score_unsup):
    # normal
    n_x = score_unsup['score']['n']
    n_y = conf_sup['conf']['n']

    # smura
    s_x = score_unsup['score']['s']
    s_y = conf_sup['conf']['s']
    plt.clf()
    # plt.xlim(4e-05, 4e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlim(5e-05, 1.5e-04)
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.2)
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.2)

def plot_line_on_scatter(conf_sup, score_unsup, path):
    # 996 : blue
    # 998 : orange
    color_dict = {'tnr0.987':'#2ca02c', 'tnr0.996':'#1f77b4','tnr0.998':'#ff7f0e'}

    # reading df to get slope and intercept
    sup_line_th = get_value_threshold(path)
    one_line_th, two_line_th = get_line_threshold(path)

    # manual line
    manual_line = [32000, 2.365]
    sup_line = [0, sup_line_th['tnr0.987']]
    plot_scatter(conf_sup, score_unsup)
    plot_one_line(manual_line, color_dict['tnr0.987'])
    plot_one_line(sup_line, color_dict['tnr0.987'])
    plt.savefig(f"{path}/manual_line_scatter.png")
    plt.clf()


    # tnr 996 998 分開存
    for tnr in ['tnr0.987', 'tnr0.996','tnr0.998']:
        # sup line
        sup_line = [0, sup_line_th[tnr]]
        plot_scatter(conf_sup, score_unsup)
        plot_one_line(sup_line, color_dict[tnr])
        plt.savefig(f"{path}/sup_line_{tnr}_scatter.png")
        plt.clf()

        # one line
        one_line = [one_line_th[tnr]['m'],one_line_th[tnr]['b']]
        plot_scatter(conf_sup, score_unsup)
        plot_one_line(one_line, color_dict[tnr])
        plt.savefig(f"{path}/one_line_{tnr}_scatter.png")
        plt.clf()

        # two line
        two_line = [two_line_th[tnr]['m'],two_line_th[tnr]['b'],two_line_th[tnr]['x']]
        plot_scatter(conf_sup, score_unsup)
        plot_two_line(two_line, color_dict[tnr])
        plt.savefig(f"{path}/two_line_{tnr}_scatter.png")
        plt.clf()

    # tnr 990 996 998 畫一起
    # sup line
    tnr987_sup_line = [0 ,sup_line_th['tnr0.987']]
    tnr996_sup_line = [0 ,sup_line_th['tnr0.996']]
    tnr998_sup_line = [0 ,sup_line_th['tnr0.998']]
    plot_scatter(conf_sup, score_unsup)
    plot_one_line(tnr987_sup_line, color_dict['tnr0.987'])
    plot_one_line(tnr996_sup_line, color_dict['tnr0.996'])
    plot_one_line(tnr998_sup_line, color_dict['tnr0.998'])
    plt.savefig(f"{path}/sup_line_all_scatter.png")
    plt.clf()

    # one line
    tnr987_one_line = [one_line_th['tnr0.987']['m'],one_line_th['tnr0.987']['b']]
    tnr996_one_line = [one_line_th['tnr0.996']['m'],one_line_th['tnr0.996']['b']]
    tnr998_one_line = [one_line_th['tnr0.998']['m'],one_line_th['tnr0.998']['b']]
    plot_scatter(conf_sup, score_unsup)
    plot_one_line(tnr987_one_line, color_dict['tnr0.987'])
    plot_one_line(tnr996_one_line, color_dict['tnr0.996'])
    plot_one_line(tnr998_one_line, color_dict['tnr0.998'])
    plt.savefig(f"{path}/one_line_all_scatter.png")
    plt.clf()

    # two line
    tnr987_two_line = [two_line_th['tnr0.987']['m'],two_line_th['tnr0.987']['b'],two_line_th['tnr0.987']['x']]
    tnr996_two_line = [two_line_th['tnr0.996']['m'],two_line_th['tnr0.996']['b'],two_line_th['tnr0.996']['x']]
    tnr998_two_line = [two_line_th['tnr0.998']['m'],two_line_th['tnr0.998']['b'],two_line_th['tnr0.998']['x']]
    plot_scatter(conf_sup, score_unsup)
    plot_two_line(tnr987_two_line, color_dict['tnr0.987'])
    plot_two_line(tnr996_two_line, color_dict['tnr0.996'])
    plot_two_line(tnr998_two_line, color_dict['tnr0.998'])
    plt.savefig(f"{path}/two_line_all_scatter.png")
    plt.clf()

def plot_roc_curve(roc_auc, fpr, tpr, path, name):
    plt.clf()
    plt.plot(fpr, tpr, color='orange', label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"{path}/{name}_roc.png")
    plt.clf()

def plot_score_distribution(n_scores, s_scores, path, name):
    plt.clf()
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(1e-05, 3.5e-05)
    # plt.xlim(4.5e-05, 8e-05)
    plt.hist(n_scores, bins=50, alpha=0.5, density=True, label="normal")
    plt.hist(s_scores, bins=50, alpha=0.5, density=True, label="smura")
    plt.xlabel('Anomaly Score')
    plt.title('Score Distribution')
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_dist.png")
    plt.clf()

def plot_score_scatter(n_max, s_max, n_mean, s_mean, path, name):
    # normal
    x1 = n_max
    y1 = n_mean
    # smura
    x2 = s_max
    y2 = s_mean
    # 設定座標軸
    # normal
    plt.clf()
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_normal_scatter.png")
    plt.clf()
    # smura
    plt.clf()
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_smura_scatter.png")
    plt.clf()
    # all
    plt.clf()
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.5, label="normal")
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.5, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}__scatter.png")
    plt.clf()

def plot_sup_unsup_scatter(conf_sup, score_unsup, path, name):
    # normal
    n_x = score_unsup['score']['n']
    n_y = conf_sup['conf']['n']

    # smura
    s_x = score_unsup['score']['s']
    s_y = conf_sup['conf']['s']

    # 設定座標軸
    # normal
    plt.clf()
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.5)
    plt.savefig(f"{path}/{name}_normal_scatter.png")
    plt.clf()
    # smura
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.5)
    plt.savefig(f"{path}/{name}_smura_scatter.png")
    plt.clf()
    # Both
    # plt.xlim(4e-05, 4e-04)
    plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.2)
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.2)
    plt.savefig(f"{path}/{name}_all_scatter.png")
    plt.clf()


