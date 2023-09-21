import time
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from options.test_options import TestOptions

from util.utils_hybird import plot_line, plot_conf_score_scatter, hybrid_one_line
                              
def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    return opt, opt.gpu_ids[0]
  
def gen_report(conf_sup, score_unsup, path, name): 
    print(path)
    
    plot_conf_score_scatter(conf_sup, score_unsup, path, name)

    # ===== Auto find threshold line =====
    one_res, model_report, one_line_time = hybrid_one_line(conf_sup, score_unsup, path)
    model_report.to_csv(os.path.join(path, f"{name}_th.csv"))
    log_name = os.path.join(path, f'{name}_report.txt')
    msg = ''
    with open(log_name, "w") as log_file:
        msg += f"=============== one line ===================\n"
        msg += f"time cost: {one_line_time}\n"
        msg += f"line equation param: m={one_res['tnr0.987_m'][0]}, b={one_res['tnr0.987_b'][0]}\n"
        msg += f"tnr0.987 tnr: {one_res['tnr0.987_tnr'][0]}\n"
        msg += f"tnr0.987 recall: {one_res['tnr0.987_recall'][0]}\n"
        msg += f"tnr0.987 precision: {one_res['tnr0.987_precision'][0]}\n"
        log_file.write(msg)

    plot_line(conf_sup, score_unsup, path, one_res['tnr0.987_m'][0], one_res['tnr0.987_b'][0])

def load_res(opt):
    res_sup = defaultdict(dict)
    for l in ['conf','label','files']:
        for t in ['n','s']:
            res_sup[l][t] = None

    res_unsup = defaultdict(dict)
    for l in ['score','label','files','all']:
        for t in ['n','s']:
            res_unsup[l][t] = None

    sup_df = pd.read_csv(opt.sup_conf_score)
    unsup_df = pd.read_csv(os.path.join(opt.unsup_ano_score, 'unsup_score_mean.csv'))
    merge_df = sup_df.merge(unsup_df, left_on='name', right_on='name')
    
    normal_filter = (merge_df['label_x']==0) & (merge_df['label_y']==0)
    smura_filter = (merge_df['label_x']==1) & (merge_df['label_y']==1)
    # print(merge_df)
    
    for l, c in zip(['conf','label','files'],['conf','label_x','name']):
        for t, f in zip(['n', 's'],[normal_filter,smura_filter]):
            res_sup[l][t] = merge_df[c][f].tolist()
    

    for l, c in zip(['score','label','files'],['score_mean','label_y','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res_unsup[l][t] = np.array(merge_df[c][f].tolist())
    
    assert res_sup['files']['n'][0] == res_unsup['files']['n'][0]
    
    # all_df = pd.read_csv(os.path.join(opt.unsup_ano_score, 'unsup_score_all.csv'))
    # normal_filter = all_df['label']==0
    # smura_filter = all_df['label']==1
    # res_unsup['all']['n'] = np.array(all_df['score'][normal_filter].tolist())
    # res_unsup['all']['s'] = np.array(all_df['score'][smura_filter].tolist())

    return res_sup, res_unsup

if __name__ == '__main__':
  
    opt, _ = initail_setting()  
    
    res_sup, res_unsup = load_res(opt)
    
    gen_report(res_sup, res_unsup, opt.results_dir, 'hybrid')
    