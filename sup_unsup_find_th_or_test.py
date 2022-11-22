import time
import os
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model

from util.utils_howard import mkdir, minmax_scaling, \
                              get_data_info, make_test_dataloader, evaluate, get_line_threshold, \
                              plot_score_distribution, plot_sup_unsup_scatter, plot_line_on_scatter, \
                              sup_unsup_prediction_spec_th, sup_unsup_prediction_spec_multi_th, \
                              sup_unsup_prediction_auto_th, sup_unsup_prediction_auto_multi_th, sup_unsup_svm, \
                              sup_prediction_spec_th, get_value_threshold, find_sup_th
import matplotlib.pyplot as plt

def initail_setting():
  opt = TestOptions().parse()
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batchSize = 1  # test code only supports batchSize = 1
  opt.serial_batches = True  # no shuffle
  opt.no_flip = True  # no flip
  opt.display_id = -1 # no visdom display
  opt.loadSize = opt.fineSize  # Do not scale!
  opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_SEResNeXt101_d23/{opt.data_version}/{opt.measure_mode}"

  mkdir(opt.results_dir)

  return opt, opt.gpu_ids[0]
  
def count_data_version(large_smura_name, small_smura_name, csv_path):
    df_large_name = pd.DataFrame(large_smura_name, columns=['name'])
    print(df_large_name)
    df_small_name = pd.DataFrame(small_smura_name, columns=['name'])
    print(df_small_name)

    df_data = pd.read_csv(csv_path)

    df_merge_large = df_data.merge(df_large_name, left_on='name', right_on='name')
    dict_large = df_merge_large['batch'].value_counts().to_dict()
    df_merge_small = df_data.merge(df_small_name, left_on='name', right_on='name')
    dict_small = df_merge_small['batch'].value_counts().to_dict()
    
    keys = dict_large.keys()
    values = dict_large.values()
    plt.clf()

    plt.bar(keys, values)
    plt.savefig(f"./large.png")
    plt.clf()

    keys = dict_small.keys()
    values = dict_small.values()
    
    plt.bar(keys, values)
    plt.savefig(f"./small.png")
    plt.clf()

def show_and_save_result(conf_sup, score_unsup, use_th, path, name):
    all_conf_sup = np.concatenate([conf_sup['conf']['n'], conf_sup['conf']['s']])
    all_score_unsup = np.concatenate([score_unsup['score']['n'], score_unsup['score']['s']])

    true_label = np.concatenate([conf_sup['label']['n'], conf_sup['label']['s']])

    plot_score_distribution(score_unsup['score']['n'], score_unsup['score']['s'], path, f"{name}_unsup")
    plot_sup_unsup_scatter(conf_sup, score_unsup, path, name)
    
    if use_th:
        # ===== blind test =====
        value_th = get_value_threshold(path)
        one_line_th, two_line_th = get_line_threshold(path)
        
        log_name = os.path.join(path, f'{result_name}_blind_test_result_log.txt')
        msg = ''
        with open(log_name, "w") as log_file:
            msg += f"=============== supervised ===================\n"
            msg += sup_prediction_spec_th(true_label, all_conf_sup, value_th, path)
            msg += f"=============== unsupervised ===================\n"
            msg += f"Normal mean: {score_unsup['all']['n'].mean()}\n"
            msg += f"Normal std: {score_unsup['all']['n'].std()}\n"
            msg += f"Smura mean: {score_unsup['all']['s'].mean()}\n"
            msg += f"Smura std: {score_unsup['all']['s'].std()}\n"
            msg += f"=============== Combine both one line ===================\n"
            msg += sup_unsup_prediction_spec_th(true_label, all_conf_sup, all_score_unsup, one_line_th, path)
            msg += f"=============== Combine both two lines ===================\n"
            msg += sup_unsup_prediction_spec_multi_th(true_label, all_conf_sup, all_score_unsup, two_line_th, path)
            
            log_file.write(msg)
    else:
        sup_res = find_sup_th(conf_sup, path)
        # ===== Auto find threshold line =====
        one_res, one_line_time = sup_unsup_prediction_auto_th(true_label, all_conf_sup, all_score_unsup, path)
        two_res, two_line_time = sup_unsup_prediction_auto_multi_th(true_label, all_conf_sup, all_score_unsup, path)
        sup_unsup_svm(true_label, all_conf_sup, all_score_unsup, path)
        log_name = os.path.join(path, f'{result_name}_find_th_log.txt')
        msg = ''
        with open(log_name, "w") as log_file:
            msg += f"=============== supervised ===================\n"
            msg += f"tnr0.987 recall: {sup_res['tnr0.987_recall']}\n"
            msg += f"tnr0.987 precision: {sup_res['tnr0.987_precision']}\n"
            msg += f"tnr0.996 recall: {sup_res['tnr0.996_recall']}\n"
            msg += f"tnr0.996 precision: {sup_res['tnr0.996_precision']}\n"
            msg += f"tnr0.998 recall: {sup_res['tnr0.998_recall']}\n"
            msg += f"tnr0.998 precision: {sup_res['tnr0.998_precision']}\n"
            msg += f"=============== one line ===================\n"
            msg += f"one line time: {one_line_time}\n"
            msg += f"tnr0.987 recall: {one_res['tnr0.987_recall']}\n"
            msg += f"tnr0.987 precision: {one_res['tnr0.987_precision']}\n"
            msg += f"tnr0.996 recall: {one_res['tnr0.996_recall']}\n"
            msg += f"tnr0.996 precision: {one_res['tnr0.996_precision']}\n"
            msg += f"tnr0.998 recall: {one_res['tnr0.998_recall']}\n"
            msg += f"tnr0.998 precision: {one_res['tnr0.998_precision']}\n"
            msg += f"=============== two line ===================\n"
            msg += f"two line time: {two_line_time}\n"
            msg += f"tnr0.987 recall: {two_res['tnr0.987_recall']}\n"
            msg += f"tnr0.987 precision: {two_res['tnr0.987_precision']}\n"
            msg += f"tnr0.996 recall: {two_res['tnr0.996_recall']}\n"
            msg += f"tnr0.996 precision: {two_res['tnr0.996_precision']}\n"
            msg += f"tnr0.998 recall: {two_res['tnr0.998_recall']}\n"
            msg += f"tnr0.998 precision: {two_res['tnr0.998_precision']}\n"
            log_file.write(msg)

    plot_line_on_scatter(conf_sup, score_unsup, path)

def model_prediction_using_record(opt):
    res_sup = defaultdict(dict)
    for l in ['conf','label','files']:
        for t in ['n','s']:
            res_sup[l][t] = None

    res_unsup = defaultdict(dict)
    for l in ['score','label','files','all']:
        for t in ['n','s']:
            res_unsup[l][t] = None

    sup_df = pd.read_csv(os.path.join(opt.results_dir, 'sup_conf.csv'))
    unsup_df = pd.read_csv(os.path.join(opt.results_dir, 'unsup_score_mean.csv'))
    merge_df = sup_df.merge(unsup_df, left_on='name', right_on='name')
    
    normal_filter = (merge_df['label_x']==0) & (merge_df['label_y']==0)
    smura_filter = (merge_df['label_x']==1) & (merge_df['label_y']==1)
    print(merge_df)
    
    for l, c in zip(['conf','label','files'],['conf','label_x','name']):
        for t, f in zip(['n', 's'],[normal_filter,smura_filter]):
            res_sup[l][t] = merge_df[c][f].tolist()
    print(res_sup['files']['n'][:10])

    for l, c in zip(['score','label','files'],['score_mean','label_y','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res_unsup[l][t] = merge_df[c][f].tolist()
    print(res_unsup['files']['n'][:10])
    
    all_df = pd.read_csv(os.path.join(opt.results_dir, 'unsup_score_all.csv'))
    normal_filter = (all_df['label']==0)
    smura_filter = (all_df['label']==1)
    res_unsup['all']['n'] = np.array(all_df['score'][normal_filter].tolist())
    res_unsup['all']['s'] = np.array(all_df['score'][smura_filter].tolist())

    return res_sup, res_unsup

if __name__ == '__main__':
  
    opt, _ = initail_setting()  

    res_sup, res_unsup = model_prediction_using_record(opt)

    result_name = f"{opt.measure_mode}_SEResNeXt101"
    show_and_save_result(res_sup, res_unsup, opt.using_threshold, opt.results_dir, result_name)
    