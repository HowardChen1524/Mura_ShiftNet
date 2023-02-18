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

from util.utils_howard import mkdir, \
                              get_data_info, make_test_dataloader, evaluate, set_seed

import torchvision.models as models
import pretrainedmodels
import timm

def initail_setting():
  opt = TestOptions().parse()
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batchSize = 1  # test code only supports batchSize = 1
  opt.serial_batches = True  # no shuffle
  opt.display_id = -1 # no visdom display
  opt.results_dir = f"/home/ldap/sallylin/Howard/Mura_ShiftNet/exp_result/Supervised/{opt.sup_model_version}/{opt.data_version}/{opt.sup_model_path.split('/')[-1][:-3]}"
  mkdir(opt.results_dir)
  set_seed(2022)
  
  return opt, opt.gpu_ids[0]
  
def export_conf(conf_sup, path, name):
  sup_name = conf_sup['files_res']['all']
  sup_conf = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
  sup_label = [0]*len(conf_sup['preds_res']['n'])+[1]*len(conf_sup['preds_res']['s'])
  df_sup = pd.DataFrame(list(zip(sup_name,sup_conf,sup_label)), columns=['name', 'conf', 'label'])
  df_sup.to_csv(os.path.join(path, f'{name}.csv'))

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
          print(x)
          
          ds_sup[x][y] = get_data_info(x, label, image_info, opt.data_dir, opt.csv_path)

  dataloaders = make_test_dataloader(ds_sup)
  # read model
  # seresnext101
  model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
  model_sup.fc = nn.Sequential(
          nn.Linear(2048, 1),
          nn.Sigmoid()
      )
  print(opt.sup_model_path)
  model_sup.load_state_dict(torch.load(opt.sup_model_path, map_location=torch.device(f"cuda:{gpu}")))  
  
  return evaluate(model_sup, dataloaders, opt.results_dir)

if __name__ == '__main__':
  
  opt, gpu = initail_setting()  
  
  # ===== supervised =====
  res_sup = supervised_model_prediction(opt, gpu)
  
  # ===== unsupervised =====

  export_conf(res_sup, opt.results_dir, 'sup_conf') # 記錄下來，防止每次都要重跑
  