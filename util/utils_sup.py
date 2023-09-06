import os
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from data.AI9_dataset import AI9_Dataset, data_transforms

# ===== Dataset & Dataloader =====
def get_data_info(t, l, image_info, data_dir):
    res = []
    image_info = image_info[(image_info["train_type"] == t) & (image_info["label"] == l) & (image_info["PRODUCT_CODE"] == "T850MVR05")]
    # image_info = image_info[(image_info["batch"] >= 24) & (image_info["batch"] <= 25) & (image_info["label"] == l) & (image_info["PRODUCT_CODE"] == "T850MVR05")]
    # print(image_info)
    
    for path, img, label, JND in zip(image_info["path"],image_info["name"],image_info["label"],image_info["MULTI_JND"]):
        img_path = os.path.join(os.path.join(data_dir), path, img)
        print(img_path)
        
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

def make_single_dataloader(ds):
    dataloader = DataLoader(ds, 
                            batch_size=1,
                            shuffle=False, 
                            num_workers=0,
                           )
    return dataloader

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

# ===== eval =====
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
    
    plot_roc_curve(res['labels_res']['all'], res['preds_res']['all'], save_path, "sup")
    print("roc curve saved!")

    return res

def export_conf(conf_sup, path, name):
  sup_name = conf_sup['files_res']['all']
  sup_conf = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
  sup_label = [0]*len(conf_sup['preds_res']['n'])+[1]*len(conf_sup['preds_res']['s'])
  df_sup = pd.DataFrame(list(zip(sup_name,sup_conf,sup_label)), columns=['name', 'conf', 'label'])
  df_sup.to_csv(os.path.join(path, f'{name}.csv'))

  print("save conf score finished!")
  
# ===== find th =====
def find_sup_th(res):
    all_label = np.concatenate([res['label']['n'], res['label']['s']])
    all_conf = np.concatenate([res['conf']['n'], res['conf']['s']])
    results = {
                'tnr0.987_th': [],
                'tnr0.987_tnr': [],
                'tnr0.987_precision': [],
                'tnr0.987_recall': [],
              }

    curve_df = get_curve_df(all_label, all_conf)

    results['tnr0.987_th'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).threshold)
    results['tnr0.987_tnr'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).tnr)
    results['tnr0.987_recall'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).recall)
    results['tnr0.987_precision'].append((curve_df[curve_df['tnr'] > 0.987].iloc[0]).precision)

    # fill empty slot
    for k, v in results.items():
      if len(v) == 0:
        results[k].append(-1)

    model_report = pd.DataFrame(results)
    
    return results, model_report

# =====
def calc_metric(labels_res, pred_res, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true=labels_res, y_pred=(pred_res >= threshold)).ravel()
    tnr = 0 if (tn + fp) == 0 else tn / (tn + fp)
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
    return threshold, tnr, precision, recall

def get_curve_df(labels_res, preds_res):
    pr_list = []
    for i in tqdm(np.linspace(min(preds_res), max(preds_res), num=10001)):
      pr_result = calc_metric(labels_res, preds_res, i)
      pr_list.append(pr_result)
    curve_df = pd.DataFrame(pr_list, columns=['threshold', 'tnr', 'precision', 'recall'])
    return curve_df

def plot_roc_curve(labels, scores, path, name):
  fpr, tpr, th = roc_curve(labels, scores)
  roc_auc = auc(fpr, tpr)

  plt.clf()
  plt.plot(fpr, tpr, color='orange', label="ROC curve (area = %0.2f)" % roc_auc)
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend(loc="lower right")
  plt.savefig(f"{path}/{name}_roc.png")
  plt.clf()



