import time
import os
from collections import defaultdict

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model

from util.utils_howard import mkdir, \
                              plot_score_distribution, plot_score_scatter, \
                              unsup_calc_metric, unsup_find_param_max_mean, set_seed
                              
import numpy as np
import pandas as pd

def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    opt.results_dir = f"{opt.results_dir}/{opt.model_version}/{opt.data_version}/{opt.measure_mode}"

    mkdir(opt.results_dir)

    set_seed(2022)

    return opt, opt.gpu_ids[0]

def export_score(score_unsup, path):
  unsup_name = score_unsup['fn']['n'] + score_unsup['fn']['s']
  unsup_label = [0]*score_unsup['mean']['n'].shape[0]+[1]*score_unsup['mean']['s'].shape[0]

  unsup_score_max = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  df_unsup_max = pd.DataFrame(list(zip(unsup_name,unsup_score_max,unsup_label)), columns=['name', 'score_max', 'label'])
  df_unsup_max.to_csv(os.path.join(path, 'unsup_score_max.csv'), index=False)

  unsup_score_mean = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  df_unsup_mean = pd.DataFrame(list(zip(unsup_name,unsup_score_mean,unsup_label)), columns=['name', 'score_mean', 'label'])
  df_unsup_mean.to_csv(os.path.join(path, 'unsup_score_mean.csv'), index=False)

  unsup_score_all = np.concatenate([score_unsup['all']['n'], score_unsup['all']['s']])
  unsup_label_all = [0]*score_unsup['all']['n'].shape[0]+[1]*score_unsup['all']['s'].shape[0]
  df_unsup_all = pd.DataFrame(list(zip(unsup_score_all,unsup_label_all)), columns=['score', 'label'])
  df_unsup_all.to_csv(os.path.join(path, 'unsup_score_all.csv'), index=False)
  print("save score finished!")

def show_and_save_result(score_unsup, path, name):
  all_max_anomaly_score = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  all_mean_anomaly_score = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])

  true_label = np.concatenate([score_unsup['label']['n'], score_unsup['label']['s']])
  
  plot_score_distribution(score_unsup['mean']['n'], score_unsup['mean']['s'], path, name)
  plot_score_scatter(score_unsup['max']['n'], score_unsup['max']['s'], score_unsup['mean']['n'], score_unsup['mean']['s'], path, name)
  
  log_name = os.path.join(path, 'result_log.txt')

  msg = ''
  with open(log_name, "w") as log_file:
    msg += f"=============== All small image mean & std =============\n" 
    msg += f"Normal mean: {score_unsup['all']['n'].mean()}\n"
    msg += f"Normal std: {score_unsup['all']['n'].std()}\n"
    msg += f"Smura mean: {score_unsup['all']['s'].mean()}\n"
    msg += f"Smura std: {score_unsup['all']['s'].std()}\n"
    msg += f"=============== Anomaly max prediction =================\n"    
    msg += unsup_calc_metric(true_label, all_max_anomaly_score, path, f"{name}_max")
    msg += f"=============== Anomaly mean prediction ================\n"
    msg += unsup_calc_metric(true_label, all_mean_anomaly_score, path, f"{name}_mean")
    msg += f"=============== Anomaly max & mean prediction ==========\n"
    msg += unsup_find_param_max_mean(true_label, all_max_anomaly_score, all_mean_anomaly_score, path, f"{name}_max_mean")
    
    log_file.write(msg)  

def model_prediction_using_record(opt):
    res_unsup = defaultdict(dict)
    for l in ['max', 'mean', 'label', 'fn']:
        for t in ['n','s']:
            res_unsup[l][t] = None

    max_df = pd.read_csv(os.path.join(opt.results_dir, 'unsup_score_max.csv'))
    mean_df = pd.read_csv(os.path.join(opt.results_dir, 'unsup_score_mean.csv'))
    merge_df = max_df.merge(mean_df, left_on='name', right_on='name')
    
    normal_filter = (merge_df['label_x']==0) & (merge_df['label_y']==0)
    smura_filter = (merge_df['label_x']==1) & (merge_df['label_y']==1)
    for l, c in zip(['max', 'mean', 'label', 'fn'],['score_max', 'score_mean', 'label_y','name']):
        for t, f in zip(['n', 's'],[normal_filter, smura_filter]):
            res_unsup[l][t] = np.array(merge_df[c][f].tolist())

    all_df = pd.read_csv(os.path.join(opt.results_dir, 'unsup_score_all.csv'))
    normal_filter = (all_df['label']==0)
    smura_filter = (all_df['label']==1)
    res_unsup['all']['n'] = np.array(all_df['score'][normal_filter].tolist())
    res_unsup['all']['s'] = np.array(all_df['score'][smura_filter].tolist())

    return res_unsup

def unsupervised_model_prediction(opt):
  res_unsup = defaultdict(dict)
  for l in ['all','max','mean', 'fn']:
    for t in ['n','s']:
      res_unsup[l][t] = None

  model = create_model(opt)
  data_loader = CreateDataLoader(opt)
  if opt.pos_normalize:
    n_all_crop_scores = []
    for i, data in enumerate(data_loader['normal']):
        print(f"img: {i}")
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)

        model.set_input(data) 
        crop_scores = model.test() # 225 張小圖的 score
    
        n_all_crop_scores.append(crop_scores)
        
    n_all_crop_scores = np.array(n_all_crop_scores)
    print(n_all_crop_scores.shape)
    n_pos_mean = np.mean(n_all_crop_scores, axis=0)
    n_pos_std = np.std(n_all_crop_scores, axis=0)

  dataset_list = [data_loader['normal'],data_loader['smura']]
  for mode, dataset in enumerate(dataset_list): 
    if mode == 0:
        opt.how_many = opt.normal_how_many
        fn_len = len(opt.testing_normal_dataroot)
    else:
        opt.how_many = opt.smura_how_many
        fn_len = len(opt.testing_smura_dataroot)
    score_log = None
    max_anomaly_score_log = None
    mean_anomaly_score_log = None
    fn_log = []
    print(f"Mode(0:normal,1:smura): {mode}, {opt.how_many}")
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        fn = data['A_paths'][0][fn_len:]
        print(f"Image num {i}: {fn}")
        fn_log.append(fn)

        # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)
        # print(data['A'].shape)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)
        # print(data['B'].shape)
        
        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)
        # print(data['M'].shape)

        # 建立 input real_A & real_B
        # it not only sets the input data with mask, but also sets the latent mask.
        model.set_input(data) 
        img_scores = model.test()
        if opt.pos_normalize:
            for pos in range(0,img_scores.shape[0]):
                img_scores[pos] = (img_scores[pos]-n_pos_mean[pos])/n_pos_std[pos]
                
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
        res_unsup['all']['n'] = score_log.copy() # all 小圖
        res_unsup['max']['n'] = max_anomaly_score_log.copy() # max
        res_unsup['mean']['n'] = mean_anomaly_score_log.copy() # mean
        res_unsup['fn']['n'] = fn_log
    else:
        res_unsup['all']['s'] = score_log.copy()
        res_unsup['max']['s'] = max_anomaly_score_log.copy()
        res_unsup['mean']['s'] = mean_anomaly_score_log.copy()
        res_unsup['fn']['s'] = fn_log
  return res_unsup

if __name__ == "__main__":

    opt, gpu = initail_setting()  

    # whether use record csv file to predict
    if opt.using_record:
        res_unsup = model_prediction_using_record(opt)
    else:
        res_unsup = unsupervised_model_prediction(opt)

    names = f"{opt.measure_mode}"
    show_and_save_result(res_unsup, opt.results_dir, names)

    if not opt.using_record:
        export_score(res_unsup, opt.results_dir)