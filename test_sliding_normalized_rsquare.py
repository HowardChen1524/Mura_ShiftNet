import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html

import numpy as np
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw

def roc(labels, scores, name):

    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]

    plot_roc_curve(fpr, tpr, name)
    
    return roc_auc, optimal_th
def plot_roc_curve(fpr, tpr, name):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name+'_roc_curve.png')
    plt.clf()
def plot_distance_distribution(n_scores, s_scores, name):
    bins = np.linspace(-1,5) # Mask MSE
    plt.hist(s_scores, bins=bins, alpha=0.5, density=True, label="smura")
    plt.hist(n_scores, bins=bins, alpha=0.5, density=True, label="normal")
    plt.xlabel('Anomaly Score')
    plt.ylabel('Probability')
    plt.title('Distribution')
    plt.legend(loc='upper right')
    plt.savefig(name + '_score.png')
    plt.clf()
def plot_distance_scatter(n_max, s_max, n_mean, s_mean, name):
    # normal
    x1 = n_max
    y1 = n_mean
    # smura
    x2 = s_max
    y2 = s_mean
    # 設定座標軸
    # normal
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.legend(loc='upper right')
    plt.savefig(name + '_normal_scatter.png')
    plt.clf()
    # smura
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(name + '_smura_scatter.png')
    plt.clf()
    # all
    plt.xlabel("max")
    plt.ylabel("mean")
    plt.title('scatter')
    plt.scatter(x1, y1, s=5, c ="blue", alpha=0.3, label="normal")
    plt.scatter(x2, y2, s=5, c ="red", alpha=0.3, label="smura")
    plt.legend(loc='upper right')
    plt.savefig(name + '_scatter.png')
    plt.clf()
def prediction(labels, scores, name):
    pred_labels = [] 
    roc_auc, optimal_th = roc(labels, scores, name)
    for score in scores:
        if score >= optimal_th:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    print("=====Prediction result=====")
    print("Confusion Matrix (row1: TN,FP | row2: FN,TP):\n", cm)
    print("AUC: ", roc_auc)
    print("TPR-FPR Threshold: ", optimal_th)
    print("Accuracy: ", (TP + TN)/DATA_NUM)
    print("Recall (TPR): ", TP/(TP+FN))
    print("TNR: ", TN/(FP+TN))
    print("PPV: ", TP/(TP+FP))
    print("NPV: ", TN/(FN+TN))
    print("False Alarm Rate (FPR): ", FP/(FP+TN))
    print("Leakage Rate (FNR): ", FN/(FN+TP))
    print("F1-Score: ", f1_score(labels, pred_labels)) # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
def max_mean_prediction(labels, max_scores, mean_scores, name):
    # score = a*max + b*mean + c
    best_a, best_b, best_c = 0, 0, 0
    best_auc = 0
    for ten_a in range(0, 10, 1):
        a = ten_a/10.0
        for ten_b in range(0, 10, 1):
            b = ten_b/10.0
            for ten_thousand_c in range(-10, 5, 1): # y -0.0001~0.00005
                c = ten_thousand_c/100000
                scores = a*max_scores + b*mean_scores + c
                fpr, tpr, th = roc_curve(labels, scores)
                current_auc = auc(fpr, tpr)
                if current_auc >= best_auc:
                    best_auc = current_auc
                    best_a = a
                    best_b = b
                    best_c = c

    print(best_auc)
    print(best_a)
    print(best_b)
    print(best_c)

    best_scores = best_a*max_scores + best_b*mean_scores + best_c
    pred_labels = [] 
    roc_auc, optimal_th = roc(labels, best_scores, name)
    for score in best_scores:
        if score >= optimal_th:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    cm = confusion_matrix(labels, pred_labels)
    TP = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[0][0]
    DATA_NUM = TN + FP + FN + TP
    print("=====Prediction result=====")
    print("Confusion Matrix (row1: TN,FP | row2: FN,TP):\n", cm)
    print("AUC: ", roc_auc)
    print("TPR-FPR Threshold: ", optimal_th)
    print("Accuracy: ", (TP + TN)/DATA_NUM)
    print("Recall (TPR): ", TP/(TP+FN))
    print("TNR: ", TN/(FP+TN))
    print("PPV: ", TP/(TP+FP))
    print("NPV: ", TN/(FN+TN))
    print("False Alarm Rate (FPR): ", FP/(FP+TN))
    print("Leakage Rate (FNR): ", FN/(FN+TP))
    print("F1-Score: ", f1_score(labels, pred_labels)) # sklearn ver: F1 = 2 * (precision * recall) / (precision + recall)
             
if __name__ == "__main__":
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    # opt.serial_batches = True  # no shuffle
    opt.serial_batches = False
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    data_loader = CreateDataLoader(opt)
    dataset_list = [data_loader['normal'],data_loader['smura']]
    
    model = create_model(opt)

    n_pos_score_log = []

    for i, data in enumerate(dataset_list[0]):
        print(f"img: {i}, {data['A_paths'][0][len(opt.testing_smura_dataroot):]}")
        
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)

        model.set_input(data) 
        crop_scores = model.test() # 225 張小圖的 score
        # print(crop_scores) # 確認位置是正確的
        
        n_pos_score_log.append(crop_scores)
        # raise
        
    # print(len(n_pos_score_log))
    # mean_list = []
    # for pos in range(len(n_pos_score_log[1])):
    #     sum = 0
    #     for id in range(len(n_pos_score_log[0])):
    #         sum += n_pos_score_log[id][pos]
    #     mean_list.append(sum/len(n_pos_score_log[0]))
    # print(np.array(mean_list))
    # print(np.array(mean_list).shape)
    # print(np.array(mean_list).mean())

    # print(n_pos_score_log)
    n_pos_score_log = np.array(n_pos_score_log)
    n_pos_mean = np.mean(n_pos_score_log, axis=0)
    # print(n_pos_mean)
    # print(n_pos_mean.shape)
    # print(n_pos_mean.mean())
    n_pos_std = np.std(n_pos_score_log, axis=0)
    n_pos_var = np.var(n_pos_score_log, axis=0)

    # raise
    # 所有小圖
    n_score_log = None
    s_score_log = None
    # 每張大圖的代表小圖
    n_max_anomaly_score_log = None
    n_mean_anomaly_score_log = None
    s_max_anomaly_score_log = None
    s_mean_anomaly_score_log = None

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

            for pos in range(0, img_scores.shape[0]):
                img_scores[pos] = (img_scores[pos]-n_pos_mean[pos])/n_pos_std[pos] # normalized

            max_anomaly_score = np.max(img_scores) # Anomaly max
            mean_anomaly_score = np.mean(img_scores) # Anomaly mean

            print(f"Max: {max_anomaly_score}")
            print(f"Mean: {mean_anomaly_score}")

            if i == 0:
                score_log = img_scores.copy()
                max_anomaly_score_log = np.array(max_anomaly_score)
                mean_anomaly_score_log = np.array(mean_anomaly_score)
            else:
                score_log = np.append(score_log, img_scores)
                max_anomaly_score_log = np.append(max_anomaly_score_log, max_anomaly_score)
                mean_anomaly_score_log = np.append(mean_anomaly_score_log, mean_anomaly_score)

        if mode == 0:
            n_score_log = score_log.copy() # all 小圖
            n_max_anomaly_score_log = max_anomaly_score_log.copy() # max
            n_mean_anomaly_score_log = mean_anomaly_score_log.copy() # mean
        else:
            s_score_log = score_log.copy()
            s_max_anomaly_score_log = max_anomaly_score_log.copy()
            s_mean_anomaly_score_log = mean_anomaly_score_log.copy()

    print(f"Normal mean: {n_pos_mean.mean()}")
    print(f"Normal std: {n_pos_std.mean()}")
    print(f"Normal var: {n_pos_var.mean()}")
    # 所有normal smura小圖的平均
    print(f"Normal mean: {n_score_log.mean()}")
    print(f"Normal std: {n_score_log.std()}")

    print(f"Smura mean: {s_score_log.mean()}")
    print(f"Smura std: {s_score_log.std()}")

    plot_distance_distribution(n_mean_anomaly_score_log, s_mean_anomaly_score_log, f"d23_8k_{opt.measure_mode}_MEAN_normalize_dist")
    plot_distance_scatter(n_max_anomaly_score_log, s_max_anomaly_score_log, 
                            n_mean_anomaly_score_log, s_mean_anomaly_score_log, f"d23_8k_{opt.measure_mode}_max_mean_normalize")
    
    all_max_anomaly_score_log = np.concatenate([n_max_anomaly_score_log, s_max_anomaly_score_log])
    all_mean_anomaly_score_log = np.concatenate([n_mean_anomaly_score_log, s_mean_anomaly_score_log])
    true_label = [0]*n_mean_anomaly_score_log.shape[0]+[1]*s_mean_anomaly_score_log.shape[0]
    
    print("=====Anomaly Score Max=====")
    prediction(true_label, all_max_anomaly_score_log, f"d23_8k_{opt.measure_mode}_MAX_normalize")
    print("=====Anomaly Score Mean=====")
    prediction(true_label, all_mean_anomaly_score_log, f"d23_8k_{opt.measure_mode}_MEAN_normalize")
    print("=====Anomaly Score Max_Mean=====")
    max_mean_prediction(true_label, all_max_anomaly_score_log, all_mean_anomaly_score_log, f"d23_8k_{opt.measure_mode}_max_mean_normalize")
    
    '''
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    for i, data in enumerate(dataset):
        # 超過設定的測試張數就跳出
        if i >= opt.how_many:
            break
        t1 = time.time()
        model.set_input(data) # create real_A and real_B
        model.test()
        t2 = time.time()
        # test 一張圖片的時間
        #print(t2-t1)

        visuals = model.get_current_visuals()
        img_path = model.get_image_paths() # 取得圖片路徑
        #print('process image... %s' % img_path)
        # visualizer.py
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    #webpage.save()
    '''

    '''
    for OpenCV and Mean 待改
    img_scores = [] 
    for crop_id in range(256):
        crop_data = {'A': torch.unsqueeze(data['A'][crop_id], 0), 
                        'B': torch.unsqueeze(data['B'][crop_id], 0), 
                        'M': torch.unsqueeze(data['M'][crop_id], 0), 
                        'A_paths': data['A_paths']}
        model.set_input(crop_data) 
        crop_score = model.test()
        img_scores.append(crop_score)
    anomaly_score_log.extend(img_scores)
    max_mse = np.max(img_scores)        
    print(max_mse)
    '''