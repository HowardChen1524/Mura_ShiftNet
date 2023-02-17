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
    opt.display_id = -1 # no visdom display

    opt.results_dir = f"{opt.results_dir}/{opt.model_version}/{opt.data_version}/{opt.measure_mode}"
    if opt.pos_normalize:
        opt.results_dir = f"{opt.results_dir}_pn"

    mkdir(opt.results_dir)

    set_seed(2022)

    return opt, opt.gpu_ids[0]

def unsupervised_model_prediction(opt):
  model = create_model(opt)
  data_loader = CreateDataLoader(opt)

  dataset_list = [data_loader['normal'],data_loader['smura']]
  for mode, dataset in enumerate(dataset_list): 
    if mode == 0:
        opt.how_many = opt.normal_how_many
        fn_len = len(opt.testing_normal_dataroot)
    else:
        opt.how_many = opt.smura_how_many
        fn_len = len(opt.testing_smura_dataroot)

    fn_log = []
    model_pred_t_list = []
    combine_t_list = []
    denoise_t_list = []
    export_t_list = []
    all_t_list = []
    print(f"Mode(0:normal,1:smura): {mode}, {opt.how_many}")
    for i, data in enumerate(dataset):
        all_start_time = time.time()
        if i >= opt.how_many:
            break
        fn = data['A_paths'][0][fn_len:]
        print(f"Image num {i}: {fn}")
        fn_log.append(fn)

        # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)
        
        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)
        # 329
        # 建立 input real_A & real_B
        # it not only sets the input data with mask, but also sets the latent mask.
        model.set_input(data)
        t = model.forward(mode, fn)
        all_t_list.append(time.time() - all_start_time)
        model_pred_t_list.append(t[0])
        combine_t_list.append(t[1])
        denoise_t_list.append(t[2])
        export_t_list.append(t[3])
        
    print(f"model_pred time cost mean: {np.mean(model_pred_t_list)}")
    print(f"combine time cost mean: {np.mean(combine_t_list)}")
    print(f"denoise time cost mean: {np.mean(denoise_t_list)}")
    print(f"export time cost mean: {np.mean(export_t_list)}")
    print(f"all time cost mean: {np.mean(all_t_list)}")

if __name__ == "__main__":

    opt, gpu = initail_setting()  
    
    unsupervised_model_prediction(opt)