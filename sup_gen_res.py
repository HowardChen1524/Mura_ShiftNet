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
                              get_data_info, make_test_dataloader, evaluate, get_line_threshold, \
                              plot_score_distribution, plot_sup_unsup_scatter, plot_line_on_scatter, \
                              sup_unsup_prediction_spec_th, sup_unsup_prediction_spec_multi_th, \
                              sup_unsup_prediction_auto_th, sup_unsup_prediction_auto_multi_th, sup_unsup_svm, \
                              sup_prediction_spec_th, get_value_threshold, set_seed

import torchvision.models as models
import pretrainedmodels
import timm

def initail_setting():
  opt = TestOptions().parse()
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batchSize = 1  # test code only supports batchSize = 1
  opt.serial_batches = True  # no shuffle
  opt.no_flip = True  # no flip
  opt.display_id = -1 # no visdom display
  opt.loadSize = opt.fineSize  # Do not scale!
  
  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_SEResNeXt101_d23/{opt.data_version}/{opt.measure_mode}"
  opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_SEResNeXt101_d23_8k/{opt.data_version}/{opt.measure_mode}_2nd"
  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_SEResNeXt101_d23_8k_RGB/{opt.data_version}/{opt.measure_mode}"
  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_SEResNeXt101_d23_8k_Gray/{opt.data_version}/{opt.measure_mode}"
  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_SEResNeXt101_d23_8k_Gray_three/{opt.data_version}/{opt.measure_mode}"

  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_ResNet50_d23_8k/{opt.data_version}/{opt.measure_mode}"
  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_Xception_d23_8k/{opt.data_version}/{opt.measure_mode}"
  # opt.results_dir = f"{opt.results_dir}/{opt.model_version}_with_ConVit_d23_8k/{opt.data_version}/{opt.measure_mode}"

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
  # model_sup.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model_sup.fc = nn.Sequential(
          nn.Linear(2048, 1),
          nn.Sigmoid()
      )
  # resnet50
  # model_sup = models.resnet50(pretrained=False)
  # model_sup.add_module("fc",
  #                 nn.Sequential(
  #                     nn.Linear(in_features=2048, out_features=1000, bias=True),
  #                     nn.ReLU(inplace=True),
  #                     nn.Dropout(p=0.3, inplace=False),
  #                     nn.Linear(in_features=1000, out_features=512, bias=True),
  #                     nn.ReLU(inplace=True),
  #                     nn.Dropout(p=0.3, inplace=False),
  #                     nn.Linear(in_features=512, out_features=64, bias=True),
  #                     nn.ReLU(),
  #                     nn.Dropout(p=0.3, inplace=False),
  #                     nn.Linear(in_features=64, out_features=1, bias=True),
  #                     nn.Sigmoid()
  #                 )
  #             )
  # xception
  # model_sup = pretrainedmodels.models.xception(pretrained=False)
  # model_sup.add_module("last_linear",
  #                 nn.Sequential(
  #                     nn.Linear(in_features=2048, out_features=1000, bias=True),
  #                     nn.ReLU(inplace=True),
  #                     nn.Dropout(p=0.3, inplace=False),
  #                     nn.Linear(in_features=1000, out_features=512, bias=True),
  #                     nn.ReLU(inplace=True),
  #                     nn.Dropout(p=0.3, inplace=False),
  #                     nn.Linear(in_features=512, out_features=64, bias=True),
  #                     nn.ReLU(),
  #                     nn.Dropout(p=0.3, inplace=False),
  #                     nn.Linear(in_features=64, out_features=1, bias=True),
  #                     nn.Sigmoid()
  #                 ))
  # convit
  # model_name = 'convit_base'
  # model_sup = timm.create_model(model_name, img_size=256, pretrained=False, num_classes=1)
  # model_sup.head = nn.Sequential(
  #     nn.Linear(in_features=768, out_features=1),
  #     nn.Sigmoid()
  # )
  print(opt.sup_model_path)
  model_sup.load_state_dict(torch.load(opt.sup_model_path, map_location=torch.device(f"cuda:{gpu}")))  
  
  return evaluate(model_sup, dataloaders, opt.results_dir)

if __name__ == '__main__':
  
  opt, gpu = initail_setting()  
  
  # ===== supervised =====
  res_sup = supervised_model_prediction(opt, gpu)
  
  # ===== unsupervised =====

  export_conf(res_sup, opt.results_dir, 'sup_conf') # 記錄下來，防止每次都要重跑
  