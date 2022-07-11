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

def roc(labels, scores, name_model):

    fpr, tpr, th = roc_curve(labels, scores)
    
    roc_auc = auc(fpr, tpr)
    
    optimal_th_index = np.argmax(tpr - fpr)
    optimal_th = th[optimal_th_index]
    
    plot_roc_curve(fpr, tpr, name_model)
    
    return roc_auc, optimal_th

def plot_distance_distribution(n_mse_log, s_mse_log, name_dist):
    bins = np.linspace(0.00001,0.0001)
    plt.hist(s_mse_log, bins, alpha=0.5, label="smura")
    plt.hist(n_mse_log, bins, alpha=0.5, label="normal")
    plt.xlabel('Distance or Score')
    plt.ylabel('Number')
    plt.title('Distribution')
    plt.legend(loc='upper right')
    plt.savefig(name_dist + '_dist.png')
    plt.show()
    plt.clf()

def plot_roc_curve(fpr, tpr, name_model):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(name_model+'_roc_curve.png')
    plt.show()
    plt.clf()

def prediction(labels, scores, name_model, normal_max):
    pred_labels = [] 
    # print(normal_max)
    roc_auc, optimal_th = roc(labels, scores, name_model)
    for score in scores:
        if score >= optimal_th:
        #if score >= normal_max:
            pred_labels.append(1)
        else:
            pred_labels.append(0)
    
    # cm = tf.math.confusion_matrix(labels=labels, predictions=pred).numpy()
    # TP = cm[1][1]
    # FP = cm[0][1]
    # FN = cm[1][0]
    # TN = cm[0][0]
    # print(cm)

    # diagonal_sum = cm.trace()
    # sum_of_all_elements = cm.sum()
    
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
    print("N_Max Threshold: ", normal_max)
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
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1 # no visdom display
    opt.loadSize = opt.fineSize  # Do not scale!

    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    data_loader = CreateDataLoader(opt)
    dataset_list = list()
    for mode in range(len(data_loader)):
        # print(mode)
        dataset_list.append(data_loader[mode].load_data())
    
    for mode in range(len(data_loader)):
        # print(mode)
        print('#testing images = %d' % len(data_loader[mode]))
    
    model = create_model(opt)

    n_score_log = list()
    n_mask_score_log = list()
    s_score_log = list()
    s_mask_score_log = list()

    # test

    for mode, dataset in enumerate(dataset_list): 
        
        if mode == 0:
            opt.how_many = opt.normal_how_many
        else:
            opt.how_many = opt.smura_how_many
        
        all_crop_scores = []

        print(f"Mode(0:normal,1:smura): {mode}, {opt.how_many}")
        for i, data in enumerate(dataset):
            # 超過設定的測試張數就跳出
            if i >= opt.how_many:
                break
            t1 = time.time()
            
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


            # for OpenCV and Mean 待改
            # crop_scores = [] 
            # for crop_id in range(256):
            #     crop_data = {'A': torch.unsqueeze(data['A'][crop_id], 0), 
            #                  'B': torch.unsqueeze(data['B'][crop_id], 0), 
            #                  'M': torch.unsqueeze(data['M'][crop_id], 0), 
            #                  'A_paths': data['A_paths']}
            #     model.set_input(crop_data) 
            #     crop_score = model.test()
            #     crop_scores.append(crop_score)

            # all_crop_scores.extend(crop_scores)
            # max_mse = np.max(crop_scores)        
            # print(max_mse)

            # 建立 input real_A & real_B
            # it not only sets the input data with mask, but also sets the latent mask.
            model.set_input(data) 
            crop_scores = model.test()
            max_crop_score = np.max(crop_scores)
            print(max_crop_score)
            
            if i == 0:
                all_crop_scores = crop_scores.copy()
            else:
                all_crop_scores.append(crop_scores)
            
            t2 = time.time()
            # test 一張大圖的時間
            # print(t2-t1)
            raise
            if mode == 0:
                n_score_log.append(max_crop_score)
            else:
                s_score_log.append(max_crop_score)

        # 計算所有圖片平均
        # if mode == 0:
        #     n_mean = np.array(all_crop_scores).mean()
        #     n_std = np.array(all_crop_scores).std()
        # else:
        #     s_mean = np.array(all_crop_scores).mean()
        #     s_std = np.array(all_crop_scores).std()

    print(f"Normal mean: {n_mean}")
    print(f"Normal std: {n_std}")

    print(f"Smura mean: {s_mean}")
    print(f"Smura std: {s_std}")

    n_score_arr = np.array(n_score_log)
    s_score_arr = np.array(s_score_log)
    del n_score_log, s_score_log

    scores = np.concatenate([n_score_arr, s_score_arr])
    true_label = [0]*n_score_arr.shape[0]+[1]*s_score_arr.shape[0]

    plot_distance_distribution(n_score_arr, s_score_arr, "Distance")

    prediction(true_label, scores, "Distance", np.max(n_score_arr))

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