import time
import os
from collections import defaultdict

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model

from util.utils_howard_old import mkdir, minmax_scaling, \
                              plot_score_distribution, plot_score_scatter, \
                              unsup_calc_metric, unsup_find_param_max_mean
                              
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def export_score(score_max, score_mean, path):
  log_name = os.path.join(path, 'score_max_log.txt')
  np.savetxt(log_name, score_max, delimiter=",")
  log_name = os.path.join(path, 'score_mean_log.txt')
  np.savetxt(log_name, score_mean, delimiter=",")

  print("save score finished!")

def show_and_save_result(score_unsup, minmax, path, name):
  all_max_anomaly_score = np.concatenate([score_unsup['max']['n'], score_unsup['max']['s']])
  all_mean_anomaly_score = np.concatenate([score_unsup['mean']['n'], score_unsup['mean']['s']])
  true_label = [0]*score_unsup['mean']['n'].shape[0]+[1]*score_unsup['mean']['s'].shape[0]
  
  export_score(all_max_anomaly_score, all_mean_anomaly_score, path)

  if minmax:
    all_max_anomaly_score = minmax_scaling(all_max_anomaly_score)
    score_unsup['max']['n'] =  all_max_anomaly_score[:score_unsup['max']['n'].shape[0]]
    score_unsup['max']['s'] =  all_max_anomaly_score[score_unsup['max']['n'].shape[0]:]

    all_mean_anomaly_score = minmax_scaling(all_mean_anomaly_score)
    score_unsup['mean']['n'] =  all_mean_anomaly_score[:score_unsup['mean']['n'].shape[0]]
    score_unsup['mean']['s'] =  all_mean_anomaly_score[score_unsup['mean']['n'].shape[0]:]

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
 
if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    # opt.serial_batches = False
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!
    if opt.pos_normalize:
        opt.result_dir = f"./exp_result/{opt.inpainting_mode}_SSIM_d23_8k/{opt.measure_mode}_pn"
    else:
        opt.result_dir = f"./exp_result/{opt.inpainting_mode}_SSIM_d23_8k/{opt.measure_mode}"
    mkdir(opt.result_dir)

    data_loader = CreateDataLoader(opt)
    
    model = create_model(opt)

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
    
    res_unsup = defaultdict(dict)

    dataset_list = [data_loader['normal'],data_loader['smura']]    
    for mode, dataset in enumerate(dataset_list): 
        
        if mode == 0:
            opt.how_many = opt.normal_how_many
        else:
            opt.how_many = opt.smura_how_many
        
        score_log = None
        max_anomaly_score_log = None
        mean_anomaly_score_log = None
        print(f"Mode(0:normal,1:smura): {mode}, {opt.how_many}")
        for i, data in enumerate(dataset):
            print(f"Image num: {i}")
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
        else:
            res_unsup['all']['s'] = score_log.copy()
            res_unsup['max']['s'] = max_anomaly_score_log.copy()
            res_unsup['mean']['s'] = mean_anomaly_score_log.copy()
    
    names = f"{opt.inpainting_mode}_{opt.measure_mode}"
    show_and_save_result(res_unsup, opt.minmax, opt.result_dir, names)