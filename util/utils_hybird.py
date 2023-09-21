import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def hybrid_one_line(conf, score, path):
    
    all_conf_sup = np.concatenate([conf['conf']['n'], conf['conf']['s']])
    all_score_unsup = np.concatenate([score['score']['n'], score['score']['s']])

    all_label = np.concatenate([conf['label']['n'], conf['label']['s']])

    results = {
        'tnr0.987_m': [],
        'tnr0.987_b': [],
        'tnr0.987_tnr': [],
        'tnr0.987_precision': [],
        'tnr0.987_recall': [],
        }
    
    curve_df, time_cost = get_curve_df(all_label, all_conf_sup, all_score_unsup)
    
    tnr987_best_recall_pos = curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].recall.argmax()
    
    results['tnr0.987_m'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).m)
    results['tnr0.987_b'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).b)
    results['tnr0.987_tnr'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).tnr)
    results['tnr0.987_recall'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).recall)
    results['tnr0.987_precision'].append((curve_df[(curve_df['tnr'] > 0.987) & (curve_df['tnr'] < 0.988)].iloc[tnr987_best_recall_pos]).precision)
    
    # fill empty slot
    for k, v in results.items():
        if len(v) == 0:
            results[k].append(-1)
    
    model_report = pd.DataFrame(results)
    # model_report.to_csv(os.path.join(path, "model_report.csv"))

    print("model report record finished!")
    return results, model_report, time_cost

def get_curve_df(labels_res, conf_res, score_res):
    # line mx + y = b
    # y = -mx + b
    start_time = time.time()
    pr_list = []
    for times_m in range(0, 1001, 1):
        m = (1000)*times_m # MSE
        for times_b in range(0, 1001, 1):
            b = (0.01)*times_b
            print(f"{m}, {b}")
            pr_result = calc_metric(labels_res, score_res, conf_res, m, b)
            pr_list.append(pr_result)
    
    curve_df = pd.DataFrame(pr_list, columns=['m', 'b', 'tnr', 'precision', 'recall'])
    time_cost = time.time() - start_time
    return curve_df, time_cost

def calc_metric(labels_res, score_res, conf_res, m, b):
    combined_scores = m*score_res + conf_res
    tn, fp, fn, tp = confusion_matrix(y_true=labels_res, y_pred=(combined_scores >= b)).ravel()
    tnr = 0 if (tn + fp) == 0 else tn / (tn + fp)
    precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
    recall = 0 if (tp + fn) == 0 else tp / (tp + fn)    

    return m, b, tnr, precision, recall  

def plot_conf_score_scatter(conf, score, path, name):
    # normal
    n_x = score['score']['n']
    n_y = conf['conf']['n']

    # smura
    s_x = score['score']['s']
    s_y = conf['conf']['s']

    # 設定座標軸
    # normal
    plt.clf()
    # plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    # plt.xlim(3e-05, 1.2e-04) # pennet
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.5)
    plt.savefig(f"{path}/{name}_normal_scatter.png")
    plt.clf()
    # smura
    # plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    # plt.xlim(3e-05, 1.2e-04) # pennet
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.5)
    plt.savefig(f"{path}/{name}_smura_scatter.png")
    plt.clf()
    # Both
    # plt.xlim(4e-05, 4e-04) 
    # plt.xlim(5e-05, 1.5e-04)
    # plt.xlim(4.5e-05, 8e-05)
    # plt.xlim(3e-05, 1.2e-04) # pennet
    plt.xlabel("score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('scatter')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.2)
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.2)
    plt.savefig(f"{path}/{name}_all_scatter.png")
    plt.clf()

def plot_line(conf, score, path, m, b):
    # mx + y = b 
    # y = -mx + b
    plot_scatter(conf, score)
    if m == 0:
        x_vals = [min(score), max(score)]
        y_vals = [b, b]
    else:
        x_vals = [(b-1)/m, b/m]
        y_vals = [1, 0]
    plt.plot(x_vals, y_vals, color='#2ca02c')
    plt.savefig(f"{path}/tnr_0.987_one_line.png")
    plt.clf()

def plot_scatter(conf, score):
    # normal
    n_x = score['score']['n']
    n_y = conf['conf']['n']
    # smura
    s_x = score['score']['s']
    s_y = conf['conf']['s']
    plt.clf()
    plt.xlabel("Score (Unsupervised)")
    plt.ylabel("Conf (Supervised)")
    plt.title('Hybrid')
    plt.scatter(n_x, n_y, s=5, c ="blue", alpha=0.2)
    plt.scatter(s_x, s_y, s=5, c ="red", alpha=0.2)