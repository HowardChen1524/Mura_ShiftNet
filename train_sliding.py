import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

import torch
if __name__ == "__main__":

    opt = TrainOptions().parse() # 讀取 cmd 的 train option param
    data_loader = CreateDataLoader(opt) # 建立 DataLoader 並讀取 (data_loader.py)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    
    # 建立 model
    model = create_model(opt)
    # 建立 visualizer
    visualizer = Visualizer(opt)
    # 初始化 total_steps
    total_steps = 0

    # 開始訓練
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): # opt.epoch_count default 1
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        
        for i, data in enumerate(dataset): # enumerate(dataset)每次都會讀入一個 batch 的資料
            iter_start_time = time.time()

            # 計算此 epoch 訓練的累積時間，default 每 50 張大圖 print 一次
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            # reset visualizer
            visualizer.reset()
            
            # batchSize default 1
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
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

            
            # for crop_id in range(opt.crop_image_num):
            #     crop_data = {'A': torch.unsqueeze(data['A'][crop_id], 0), 
            #                  'B': torch.unsqueeze(data['B'][crop_id], 0), 
            #                  'M': data['M'], 
            #                  'A_paths': data['A_paths']}

            # it not only sets the input data with mask, but also sets the latent mask.
            # 建立 input real_A & real_B
            model.set_input(data) 

            # 依照 display_freq 去 set_show_map_true()？
            # Additonal, should set it before 'optimize_parameters()'.
            # if total_steps % opt.display_freq == 0:
            #     if opt.show_flow: # 預設 False
            #         model.set_show_map_true()
            
            model.optimize_parameters()

            # 依照 display_freq 去 display_current_results
            # if total_steps % opt.display_freq == 0:
            #     save_result = total_steps % opt.update_html_freq == 0
            #     if opt.show_flow:
            #         model.set_flow_src()
            #         model.set_show_map_false()
            #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            
            # 計算每 50 張大圖的時間
            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize 
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            
            # 依照 save_latest_freq 去 save_networks
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                model.save_networks('latest')

            # 取得現在時間放入 iter_data_time?
            iter_data_time = time.time()
        
        # 依照 save_epoch_freq 去 save_networks
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                    (epoch, total_steps))
            model.save_networks('latest')
            if not opt.only_lastest:
                model.save_networks(epoch)

        # print 一個 epoch 所花的時間
        print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        
        model.update_learning_rate()
    