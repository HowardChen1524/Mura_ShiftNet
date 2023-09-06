import os
from collections import defaultdict

import numpy as np
import pandas as pd

from options.test_options import TestOptions

from util.utils_unsup import plot_score_distribution, find_unsup_th

def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    return opt, opt.gpu_ids[0]
  
def gen_report(score, path, name):
    
    plot_score_distribution(score['score']['n'], score['score']['s'], path, f"{name}_unsup")
    
    res, model_report = find_unsup_th(score)
    model_report.to_csv(os.path.join(path, f"{name}_th.csv"))
    # ===== Auto find threshold line =====
    log_name = os.path.join(path, f'{name}_report.txt')
    msg = ''
    with open(log_name, "w") as log_file:
        msg += f"=============== unsupervised ===================\n"
        msg += f"tnr0.987 threshold: {res['tnr0.987_th'][0]}\n"
        msg += f"tnr0.987 tnr: {res['tnr0.987_tnr'][0]}\n"
        msg += f"tnr0.987 recall: {res['tnr0.987_recall'][0]}\n"
        msg += f"tnr0.987 precision: {res['tnr0.987_precision'][0]}\n"
        log_file.write(msg)

    # plot_line_on_scatter(conf_sup, score, path)

def load_res(opt):
    res = defaultdict(dict)
    for l in ['score','label','files','all']:
        for t in ['n','s']:
            res[l][t] = None

    unsup_df = pd.read_csv(opt.unsup_ano_score)
    
    normal_filter = unsup_df['label']==0
    smura_filter = unsup_df['label']==1
    # print(unsup_df)
    
    for l, c in zip(['score','label','files'],['score_mean','label','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res[l][t] = np.array(unsup_df[c][f].tolist())
    # print(res['files']['n'][:10])
    
    return res

if __name__ == '__main__':
    opt, _ = initail_setting()  
    res = load_res(opt)
    gen_report(res, opt.results_dir, 'unsup')
    