import time
import numpy as np

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models import create_model

from util.utils_howard import mkdir, set_seed

from pytorch_grad_cam import GradCAM
import torch
import torch.nn as nn
import os
import cv2
# 同時讀圖
## create dataset
## 每次給 unsup and sup 圖
# 取得 diff value and conf value
# 依照 weight 相加
# 取前?%得出來                 


def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    opt.results_dir = f"{opt.results_dir}/{opt.data_version}/{opt.resolution}/combine_sup_unsup"

    mkdir(opt.results_dir)

    set_seed(2022)

    return opt, opt.gpu_ids[0]

def sup_init(opt):
    # load model
    model_sup = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
    model_sup.fc = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    sup_model_path = os.path.join(opt.checkpoints_dir, f"{opt.sup_model_version}/model.pt")
    print(sup_model_path)
    model_sup.load_state_dict(torch.load(sup_model_path, map_location=torch.device(f"cuda:{gpu}")))  
    cam = GradCAM(model=model_sup, target_layers=[model_sup.layers[-1]], use_cuda=True)
    return model_sup, cam

def main(opt):
    
    sup_model, gradcam = sup_init(opt)
    model = create_model(opt)

    data_loader = CreateDataLoader(opt)

    dataset = data_loader['smura']
  
    opt.how_many = opt.smura_how_many
    fn_len = len(opt.testing_smura_dataroot)

    fn_log = []
    for i, data in enumerate(dataset):
       
        if i >= opt.how_many:
            break

        fn = data['A_paths'][0][fn_len:]
        print(f"Image num {i}: {fn}")
        fn_log.append(fn)

        sup_data = data[0]
        unsup_data = data[1]
        
        # sup
        grad_conf = gradcam(input_tensor=sup_data, aug_smooth=False, eigen_smooth=False)
        # grad_conf = cv2.resize()
        
        # unsup
        bs, ncrops, c, h, w = unsup_data['A'].size()
        unsup_data['A'] = unsup_data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = unsup_data['B'].size()
        unsup_data['B'] = unsup_data['B'].view(-1, c, h, w)
        
        bs, ncrops, c, h, w = unsup_data['M'].size()
        unsup_data['M'] = unsup_data['M'].view(-1, c, h, w)
       
        # 建立 input real_A & real_B
        # it not only sets the input unsup_data with mask, but also sets the latent mask.
        model.set_input(unsup_data)
        diff_score = model.get_diff_res()
        
        alpha = opt.combine_weight
        combine_res = grad_conf*alpha + diff_score*(1-alpha)

        top_k = opt.top_k

        # filter top five percent pixel value
        num_pixels = combine_res.numel()
        num_top_pixels = int(num_pixels * top_k)
        filter, _ = combine_res.view(-1).kthvalue(num_pixels - num_top_pixels)
        combine_res[combine_res>=filter] = 1
        combine_res[combine_res<filter] = -1
        
        save_path = os.path.join(opt.results_dir, f'combine/{top_k:.3f}_diff_pos_area_{opt.min_area}/imgs')
        mkdir(save_path)             
        cv2.imwrite(os.path.join(save_path, fn), combine_res)
        
if __name__ == "__main__":

    opt, gpu = initail_setting()  
    
    main(opt)