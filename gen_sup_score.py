import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from options.test_options import TestOptions
from util.utils import set_seed
from util.utils_sup import get_data_info, make_test_dataloader, evaluate, export_conf

def initail_setting():
  opt = TestOptions().parse()
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batchSize = 1  # test code only supports batchSize = 1
  opt.serial_batches = True  # no shuffle
  set_seed(2022)
  
  return opt, opt.gpu_ids[0]

def model_prediction(opt, gpu):

  image_info = pd.read_csv(opt.data_csv_path)

  # create dataset & dataloader
  ds_sup = defaultdict(dict)
  for x in ["test"]:
      for y in ["mura", "normal"]:
          if y == "mura":
              label = 1
          elif y == "normal":
              label = 0
          ds_sup[x][y] = get_data_info(x, label, image_info, opt.sup_dataroot)

  dataloaders = make_test_dataloader(ds_sup)

  # load Model
  model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
  model_sup.fc = nn.Sequential(
          nn.Linear(2048, 1),
          nn.Sigmoid()
      )
  model_sup.load_state_dict(torch.load(f"{opt.checkpoints_dir}/{opt.sup_model_version}/model.pt", map_location=torch.device(f"cuda:{gpu}")))  
  
  return evaluate(model_sup, dataloaders, opt.results_dir)

if __name__ == '__main__':
  opt, gpu = initail_setting()    
  res = model_prediction(opt, gpu)
  export_conf(res, opt.results_dir, 'sup_conf') # 記錄下來，防止每次都要重跑
  