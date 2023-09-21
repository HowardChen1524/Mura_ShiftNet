import os
from collections import defaultdict

import numpy as np
import pandas as pd

from options.test_options import TestOptions

from util.utils_unsup import plot_score_distribution, find_unsup_th, plot_line, calc_roc_curve, plot_roc_curve

def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    return opt, opt.gpu_ids[0]
  
def gen_report(score, path, name):

    plot_score_distribution(score, path, name)
    roc_auc, optimal_th, fpr, tpr = calc_roc_curve(score) 
    plot_roc_curve(roc_auc, fpr, tpr, path, name)
    
    res, model_report = find_unsup_th(score)
    model_report.to_csv(os.path.join(path, f"{name}_th.csv"))
    # ===== Auto find threshold line =====
    log_name = os.path.join(path, f'{name}_report.txt')
    msg = ''
    with open(log_name, "w") as log_file:
        msg += f"=============== unsupervised ===================\n"
        msg += f"Normal mean: {score['all']['n'].mean()}\n"
        msg += f"Normal std: {score['all']['n'].std()}\n"
        msg += f"Smura mean: {score['all']['s'].mean()}\n"
        msg += f"Smura std: {score['all']['s'].std()}\n"
        msg += f"AUC: {roc_auc}\n"
        msg += f"Highest TPR-FPR threshold: {optimal_th}\n"
        msg += f"tnr0.987 threshold: {res['tnr0.987_th'][0]}\n"
        msg += f"tnr0.987 tnr: {res['tnr0.987_tnr'][0]}\n"
        msg += f"tnr0.987 recall: {res['tnr0.987_recall'][0]}\n"
        msg += f"tnr0.987 precision: {res['tnr0.987_precision'][0]}\n"
        log_file.write(msg)

    plot_line(score, path, res['tnr0.987_th'][0])

def load_res(opt):
    res = defaultdict(dict)
    for l in ['score','label','files','all']:
        for t in ['n','s']:
            res[l][t] = None

    unsup_df = pd.read_csv(os.path.join(opt.unsup_ano_score, 'unsup_score_mean.csv'))
    
    normal_filter = unsup_df['label']==0
    smura_filter = unsup_df['label']==1
    # print(unsup_df)
    
    for l, c in zip(['score','label','files'],['score_mean','label','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res[l][t] = np.array(unsup_df[c][f].tolist())
    # print(res['files']['n'][:10])
    
    all_df = pd.read_csv(os.path.join(opt.unsup_ano_score, 'unsup_score_all.csv'))
    normal_filter = all_df['label']==0
    smura_filter = all_df['label']==1
    res['all']['n'] = np.array(all_df['score'][normal_filter].tolist())
    res['all']['s'] = np.array(all_df['score'][smura_filter].tolist())

    return res

if __name__ == '__main__':
    opt, _ = initail_setting()  
    res = load_res(opt)
    gen_report(res, opt.results_dir, 'unsup')
    