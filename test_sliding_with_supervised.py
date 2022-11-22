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

from util.utils_howard_old import mkdir, minmax_scaling, \
                              plot_score_distribution, plot_sup_unsup_scatter, plot_line_on_scatter, \
                              get_data_info, evaluate, get_line_threshold, get_value_threshold, \
                              sup_unsup_prediction_spec_th, sup_unsup_prediction_spec_multi_th, sup_prediction_spec_th, \
                              sup_unsup_prediction_auto_th, sup_unsup_prediction_auto_multi_th, sup_unsup_svm

from sklearn.preprocessing import MinMaxScaler
from supervised_model.wei_dataloader import make_test_dataloader

def export_conf_score(conf_sup, score_unsup, path):
  log_name = os.path.join(path, 'conf_log.txt')
  np.savetxt(log_name, conf_sup, delimiter=",")
  log_name = os.path.join(path, 'score_log.txt')
  np.savetxt(log_name, score_unsup, delimiter=",")
  # df_res = pd.DataFrame(list(zip(conf_sup['preds_res']['all'], score_unsup)), columns=["conf", "score"])

  # df_res.to_csv(f"{path}/model_conf_score.csv", index=None)
  print("save conf score finished!")

def show_and_save_result(conf_sup, score_unsup, minmax, pn, ut, path, name):

    all_conf_sup = np.concatenate([conf_sup['preds_res']['n'], conf_sup['preds_res']['s']])
    if pn:
        unsup_score_type = 'max'
    else:
        unsup_score_type = 'mean'
    all_score_unsup = np.concatenate([score_unsup[unsup_score_type]['n'], score_unsup[unsup_score_type]['s']])

    true_label = [0]*score_unsup[unsup_score_type]['n'].shape[0]+[1]*score_unsup[unsup_score_type]['s'].shape[0]

    export_conf_score(all_conf_sup, all_score_unsup, path) # 記錄下來，防止每次都要重跑

    plot_score_distribution(score_unsup[unsup_score_type]['n'], score_unsup[unsup_score_type]['s'], path, f"{name}_unsup_no_minmax")
  
    if minmax:
        all_score_unsup = minmax_scaling(all_score_unsup)
        score_unsup[unsup_score_type]['n'] =  all_score_unsup[:score_unsup[unsup_score_type]['n'].shape[0]]
        score_unsup[unsup_score_type]['s'] =  all_score_unsup[score_unsup[unsup_score_type]['n'].shape[0]:]

    plot_score_distribution(conf_sup['preds_res']['n'], conf_sup['preds_res']['s'], path, f"{name}_sup")
    plot_score_distribution(score_unsup[unsup_score_type]['n'], score_unsup[unsup_score_type]['s'], path, f"{name}_unsup")
    plot_sup_unsup_scatter(conf_sup, score_unsup, path, name, unsup_score_type)
    
    if ut:
       # ===== blind test =====
        value_th = get_value_threshold(path)
        one_line_th, two_line_th = get_line_threshold(path)
        
        log_name = os.path.join(path, f'{name}_blind_test_result_log.txt')
        msg = ''
        with open(log_name, "w") as log_file:
            msg += f"=============== supervised ===================\n"
            msg += sup_prediction_spec_th(true_label, all_conf_sup, value_th, path)
            msg += f"=============== Combine both one line ===================\n"
            msg += sup_unsup_prediction_spec_th(true_label, all_conf_sup, all_score_unsup, one_line_th, path)
            msg += f"=============== Combine both two lines ===================\n"
            msg += sup_unsup_prediction_spec_multi_th(true_label, all_conf_sup, all_score_unsup, two_line_th, path)
            log_file.write(msg)
    else:
        pass
        # ===== Auto find threshold line =====
        sup_unsup_prediction_auto_th(true_label, all_conf_sup, all_score_unsup, path)
        sup_unsup_prediction_auto_multi_th(true_label, all_conf_sup, all_score_unsup, path)
        sup_unsup_svm(true_label, all_conf_sup, all_score_unsup, path)

        plot_line_on_scatter(conf_sup, score_unsup, path, unsup_score_type)

if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!
    # if opt.pos_normalize:
    #     opt.result_dir = f"./exp_result/{opt.inpainting_mode}_d23_8k_SEResNeXt101_d23_baseline/{opt.measure_mode}_pn"
    # else:
    #     opt.result_dir = f"./exp_result/{opt.inpainting_mode}_d23_8k_SEResNeXt101_d23_baseline/{opt.measure_mode}"
    if opt.pos_normalize:
        opt.result_dir = f"./exp_result/{opt.inpainting_mode}_d23_8k_SEResNeXt101_d23_baseline_d2425/{opt.measure_mode}_pn"
    else:
        opt.result_dir = f"./exp_result/{opt.inpainting_mode}_d23_8k_SEResNeXt101_d23_baseline_d2425/{opt.measure_mode}"
    mkdir(opt.result_dir)

    # Supervised
    # create dataset dataloader
    # data_dir = r'/hcds_vol/private/howard/mura_data/d23_merge/'
    # csv_path = r'/hcds_vol/private/howard/mura_data/d23_merge/data_merged.csv'
    data_dir = r'/hcds_vol/private/howard/mura_data/d25_merge/'
    csv_path = r'/hcds_vol/private/howard/mura_data/d25_merge/d25_data_merged.csv'
    image_info = pd.read_csv(csv_path)
    ds_sup = defaultdict(dict)
    for x in ["test"]:
        for y in ["mura", "normal"]:
            if y == "mura":
                label = 1
            elif y == "normal":
                label = 0
            ds_sup[x][y] = get_data_info(x, label, image_info, data_dir, csv_path)

    dataloaders = make_test_dataloader(ds_sup)
    # read model
    model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
    model_sup.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    model_sup.load_state_dict(torch.load('./supervised_model/model.pt', map_location=torch.device(f"cuda:{opt.gpu_ids[0]}")))  
    
    res_sup = evaluate(model_sup, dataloaders, opt.result_dir)
    print(res_sup['files_res']['all'])
    
    # Unsupervised
    model = create_model(opt)

    data_loader = CreateDataLoader(opt)
    
    if opt.pos_normalize:
        n_all_crop_scores = []
        s_all_crop_scores = None

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

    res_unsup = defaultdict(dict)
    for mode, dataset in enumerate(dataset_list): 
        
        if mode == 0:
            opt.how_many = opt.normal_how_many
        else:
            opt.how_many = opt.smura_how_many
        
        score_log = None
        max_anomaly_score_log = None
        mean_anomaly_score_log = None
        fn_log = []
        print(f"Mode(0:normal,1:smura): {mode}, {opt.how_many}")
        for i, data in enumerate(dataset):
            fn = data['A_paths'][0][len(opt.testing_smura_dataroot):]
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
            res_unsup['fn']['n'] = np.array(fn_log)
        else:
            res_unsup['all']['s'] = score_log.copy()
            res_unsup['max']['s'] = max_anomaly_score_log.copy()
            res_unsup['mean']['s'] = mean_anomaly_score_log.copy()
            res_unsup['fn']['s'] = np.array(fn_log)
    print(res_unsup['fn']['n'])
    print(res_unsup['fn']['s'])
    raise
    # result_name = f"{opt.inpainting_mode}_d23_8k_SEResNeXt101_d23_baseline"
    result_name = f"{opt.inpainting_mode}_d23_8k_8k_SEResNeXt101_d23_baseline_d2425"

    show_and_save_result(res_sup, res_unsup, opt.minmax, opt.pos_normalize, opt.using_threshold, opt.result_dir, result_name)