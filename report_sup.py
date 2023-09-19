import os
from collections import defaultdict

import numpy as np
import pandas as pd

from options.test_options import TestOptions
from util.utils_sup import plot_conf_distribution, find_sup_th, plot_line

def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    return opt, opt.gpu_ids[0]
  
def gen_report(conf, path, name):

    plot_conf_distribution(conf, path, name)

    res, model_report = find_sup_th(conf)
    model_report.to_csv(os.path.join(path, f"{name}_th.csv"))
    log_name = os.path.join(path, f'{name}_report.txt')
    msg = ''
    with open(log_name, "w") as log_file:
        msg += f"=============== supervised ===================\n"
        msg += f"tnr0.987 threshold: {res['tnr0.987_th'][0]}\n"
        msg += f"tnr0.987 tnr: {res['tnr0.987_tnr'][0]}\n"
        msg += f"tnr0.987 recall: {res['tnr0.987_recall'][0]}\n"
        msg += f"tnr0.987 precision: {res['tnr0.987_precision'][0]}\n"
        log_file.write(msg)

    plot_line(conf, path, res['tnr0.987_th'][0])

def load_res(opt):
    res = defaultdict(dict)
    for l in ['conf','label','files']:
        for t in ['n','s']:
            res[l][t] = None

    sup_df = pd.read_csv(opt.sup_conf_score)
    
    normal_filter = sup_df['label']==0
    smura_filter = sup_df['label']==1
    # print(sup_df)
    
    for l, c in zip(['conf','label','files'],['conf','label','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res[l][t] = np.array(sup_df[c][f].tolist())
    # print(res['files']['n'][:10])
    
    return res

if __name__ == '__main__':
  
    opt, _ = initail_setting()  
    res = load_res(opt)    
    gen_report(res, opt.results_dir, 'sup')
    