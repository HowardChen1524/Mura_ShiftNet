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
                              sup_prediction_spec_th, get_value_threshold, set_seed

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

  set_seed(2022)
  
  return opt, opt.gpu_ids[0]
  
def export_conf_score(conf_sup, score_unsup, path):
  sup_name = conf_sup['files_res']['all']
  sup_conf = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
  sup_label = [0]*len(conf_sup['preds_res']['n'])+[1]*len(conf_sup['preds_res']['s'])
  df_sup = pd.DataFrame(list(zip(sup_name,sup_conf,sup_label)), columns=['name', 'conf', 'label'])
  df_sup.to_csv(os.path.join(path, 'sup_conf.csv'))

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
  print("save conf score finished!")

def supervised_model_prediction(opt, gpu):
  image_info = pd.read_csv(opt.csv_path)
  ds_sup = defaultdict(dict)
  for x in ["test"]:
      for y in ["mura", "normal"]:
          if y == "mura":
              label = 1
          elif y == "normal":
              label = 0
          ds_sup[x][y] = get_data_info(x, label, image_info, opt.data_dir, opt.csv_path)

  dataloaders = make_test_dataloader(ds_sup)
  # read model
  model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
  model_sup.fc = nn.Sequential(
          nn.Linear(2048, 1),
          nn.Sigmoid()
      )
  model_sup.load_state_dict(torch.load(opt.sup_model_path, map_location=torch.device(f"cuda:{gpu}")))  
  
  return evaluate(model_sup, dataloaders, opt.results_dir)

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

if __name__ == '__main__':
  
  opt, gpu = initail_setting()  
  
  # ===== supervised =====
  res_sup = supervised_model_prediction(opt, gpu)
  
  # ===== unsupervised =====
  res_unsup = unsupervised_model_prediction(opt)

  export_conf_score(res_sup, res_unsup, opt.results_dir) # 記錄下來，防止每次都要重跑
  