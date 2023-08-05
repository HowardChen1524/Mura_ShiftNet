import os
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.logger import Logger
import torch
import numpy as np
import matplotlib.pyplot as plt
from util.utils_howard import mkdir

def plot_loss(epochs, loss, name):
    plt.plot(epochs, loss)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(name)
    # plt.legend(loc='upper right')
    plt.savefig(f"{opt.checkpoints_dir}/{opt.model_version}/loss_{name}.png")
    plt.clf()
    
if __name__ == "__main__":

    opt = TrainOptions().parse() # 讀取 cmd 的 train option param
    mkdir(os.path.join(opt.checkpoints_dir, opt.model_version))
    data_loader = CreateDataLoader(opt) # 建立 DataLoader 並讀取 (data_loader.py)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    
    # 建立 model
    model = create_model(opt)

    # 建立 logger
    logger = Logger(opt)

    # 初始化 total_steps
    total_steps = 0
    
    # loss list
    GAN_loss_list = []
    G_L1_loss_list = []
    D_loss_list = []
    style_loss_list = []
    content_loss_list = []
    tv_loss_list = []
    ssim_loss_list = []

    # 開始訓練
    for epoch in range(opt.epoch_count, opt.niter + 1): # opt.epoch_count default 1
        epoch_start_time = time.time()
        iter_data_time = time.time()
        
        for i, data in enumerate(dataset): # enumerate(dataset)每次都會讀入一個 mini-batch 的資料
            if i >= opt.fix_step:
                print(f'Limit Step {opt.fix_step}')
                break

            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
                        
            # batchSize default 1
            total_steps += opt.batchSize
            
            # remove extra dim
            bs, ncrops, c, h, w = data['A'].size()
            data['A'] = data['A'].view(-1, c, h, w)

            bs, ncrops, c, h, w = data['B'].size()
            data['B'] = data['B'].view(-1, c, h, w)

            bs, ncrops, c, h, w = data['M'].size()
            data['M'] = data['M'].view(-1, c, h, w)
            
            # training
            model.set_input(data)  
            model.optimize_parameters()
            
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize 
                logger.print_current_losses(epoch, total_steps, losses, t, t_data)

            # 取得現在時間放入 iter_data_time?
            iter_data_time = time.time()

        model.update_learning_rate()

        # 依照 save_epoch_freq 去 save_networks
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save_networks('latest')
            if not opt.only_lastest:
                model.save_networks(epoch)

        loss_dict = model.get_current_losses()
        GAN_loss_list.append(loss_dict['G_GAN'])
        G_L1_loss_list.append(loss_dict['G_L1'])
        D_loss_list.append(loss_dict['D'])
        style_loss_list.append(loss_dict['style'])
        content_loss_list.append(loss_dict['content'])
        tv_loss_list.append(loss_dict['tv'])
        ssim_loss_list.append(loss_dict['ssim'])

        epoch_list = np.linspace(1, epoch, epoch).astype(int)
        plot_loss(epoch_list, GAN_loss_list, 'GAN')
        plot_loss(epoch_list, G_L1_loss_list, 'L1')
        plot_loss(epoch_list, D_loss_list, 'D')
        plot_loss(epoch_list, style_loss_list, 'style')
        plot_loss(epoch_list, content_loss_list, 'content')
        plot_loss(epoch_list, tv_loss_list, 'tv')
        plot_loss(epoch_list, ssim_loss_list, 'ssim')
        
        # print 一個 epoch 所花的時間
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
    
    

    