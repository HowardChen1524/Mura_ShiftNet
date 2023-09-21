import os
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix



# ===== eval =====
def evaluate(opt, dataset_list, model):
    res = defaultdict(dict)
    for l in ['all','max','mean', 'fn', 'label']:
        for t in ['n','s']:
            res[l][t] = None

    for mode, dataset in enumerate(dataset_list): 
        if mode == 0:
            opt.how_many = opt.normal_how_many
        else:
            opt.how_many = opt.smura_how_many
        score_log = None
        max_anomaly_score_log = None
        mean_anomaly_score_log = None
        fn_log = []
        pred_time = []
        print(f"Mode(0:normal,1:smura): {mode}, {opt.how_many}")
        for i, data in enumerate(dataset):
            if i >= opt.how_many:
                break
            fn = data['A_paths'][0].split("/")[-1]
            print(f"Image num {i}: {fn}")
            fn_log.append(fn)

            # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
            bs, ncrops, c, h, w = data['A'].size()
            data['A'] = data['A'].view(-1, c, h, w)

            bs, ncrops, c, h, w = data['B'].size()
            data['B'] = data['B'].view(-1, c, h, w)
            
            bs, ncrops, c, h, w = data['M'].size()
            data['M'] = data['M'].view(-1, c, h, w)

            # 建立 input real_A & real_B
            # it not only sets the input data with mask, but also sets the latent mask.
            model.set_input(data)
            t, img_scores = model.test()
            print(f"Time: {t}")
            pred_time.append(t)
            # if opt.pos_normalize:
            #     for pos in range(0, img_scores.shape[0]):
            #         img_scores[pos] = (img_scores[pos]-n_pos_mean[pos])/n_pos_std[pos]
                    
            max_anomaly_score = np.max(img_scores) # Anomaly max
            mean_anomaly_score = np.mean(img_scores) # Anomaly mean
            print(f"{opt.measure_mode} Max: {max_anomaly_score}")
            print(f"{opt.measure_mode} Mean: {mean_anomaly_score}")

            if i == 0:
                score_log = img_scores.copy()
                max_anomaly_score_log = np.array(max_anomaly_score)
                mean_anomaly_score_log = np.array(mean_anomaly_score)
            else:
                score_log = np.append(score_log, img_scores)
                max_anomaly_score_log = np.append(max_anomaly_score_log, max_anomaly_score)
                mean_anomaly_score_log = np.append(mean_anomaly_score_log, mean_anomaly_score)
        if mode == 0:
            res['all']['n'] = score_log.copy() # all 小圖
            res['max']['n'] = max_anomaly_score_log.copy() # max
            res['mean']['n'] = mean_anomaly_score_log.copy() # mean
            res['fn']['n'] = fn_log
            res['label']['n'] = [0]*opt.normal_how_many
        else:
            res['all']['s'] = score_log.copy()
            res['max']['s'] = max_anomaly_score_log.copy()
            res['mean']['s'] = mean_anomaly_score_log.copy()
            res['fn']['s'] = fn_log
            res['label']['s'] = [1]*opt.smura_how_many
    return np.mean(pred_time), res

def export_score(score, path):
    name = score['fn']['n'] + score['fn']['s']
    label = score['label']['n'] + score['label']['s']

    score_max = np.concatenate([score['max']['n'], score['max']['s']])
    df_max = pd.DataFrame(list(zip(name,score_max,label)), columns=['name', 'score_max', 'label'])
    df_max.to_csv(os.path.join(path, 'unsup_score_max.csv'), index=False)

    score_mean = np.concatenate([score['mean']['n'], score['mean']['s']])
    df_mean = pd.DataFrame(list(zip(name,score_mean,label)), columns=['name', 'score_mean', 'label'])
    df_mean.to_csv(os.path.join(path, 'unsup_score_mean.csv'), index=False)

    score_all = np.concatenate([score['all']['n'], score['all']['s']])
    label_all = [0]*score['all']['n'].shape[0]+[1]*score['all']['s'].shape[0]
    df_all = pd.DataFrame(list(zip(score_all,label_all)), columns=['score', 'label'])
    df_all.to_csv(os.path.join(path, 'unsup_score_all.csv'), index=False)
    print("save score finished!")

# ===== find th =====
def find_unsup_th(res):
    all_label = np.concatenate([res['label']['n'], res['label']['s']])
    all_score = np.concatenate([res['score']['n'], res['score']['s']])
    
    results = {
                'tnr0.987_th': [],
                'tnr0.987_tnr': [],
                'tnr0.987_precision': [],
                'tnr0.987_recall': [],
              }

    curve_df = get_curve_df(all_label, all_score)

    tnr987_best_recall_pos = curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].recall.argmax()

    results['tnr0.987_th'].append((curve_df[curve_df['tnr'] > 0.987].iloc[tnr987_best_recall_pos]).threshold)
    results['tnr0.987_tnr'].append((curve_df[curve_df['tnr'] > 0.987].iloc[tnr987_best_recall_pos]).tnr)
    results['tnr0.987_recall'].append((curve_df[curve_df['tnr'] > 0.987].iloc[tnr987_best_recall_pos]).recall)
    results['tnr0.987_precision'].append((curve_df[curve_df['tnr'] > 0.987].iloc[tnr987_best_recall_pos]).precision)

    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)

    model_report = pd.DataFrame(results)
    
    return results, model_report
    
# =====
def calc_metric(labels_res, preds_res, threshold):
    tn, fp, fn, tp = confusion_matrix(y_true=labels_res, y_pred=(preds_res >= threshold)).ravel()
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

def calc_roc_curve(res):
    all_label = np.concatenate([res['label']['n'], res['label']['s']])
    all_score = np.concatenate([res['score']['n'], res['score']['s']])
    
    fpr, tpr, th = roc_curve(all_label, all_score)
    roc_auc = auc(fpr, tpr)

    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    return roc_auc, optimal_th, fpr, tpr

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

def plot_line(score, path, th):
    # mx + y = b 
    # y = -mx + b
    plot_scatter(score)
    
    x_vals = [th, th]
    y_vals = [0, 1]
    plt.plot(x_vals, y_vals, color='#2ca02c')
    plt.savefig(f"{path}/tnr_0.987_one_line.png")
    plt.clf()

def plot_scatter(score):
    # normal
    n_x = score['score']['n']
    n_y = [0]*len(score['score']['n'])
    # smura
    s_x = score['score']['s']
    s_y = [1]*len(score['score']['s'])
    plt.clf()
    plt.xlabel("Score (Unsupervised)")
    plt.ylabel("x")
    plt.title('Unsupervised')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.2)
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.2)

def plot_score_distribution(score, path, name):
    plt.clf()
    plt.hist(score['score']['n'], bins=50, alpha=0.5, density=True, label="normal")
    plt.hist(score['score']['s'], bins=50, alpha=0.5, density=True, label="smura")
    plt.xlabel('Anomaly Score')
    plt.title('Unsupervised')
    plt.legend(loc='upper right')
    plt.savefig(f"{path}/{name}_dist.png")


