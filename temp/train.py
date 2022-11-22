import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

if __name__ == "__main__":

    # 讀取 cmd 的 train option param
    opt = TrainOptions().parse()
    # 建立 DataLoader 並讀取 (data_loader.py)
    # isTrain 來自 train option
    data_loader = CreateDataLoader(opt)
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
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): 
        # 初始化 epoch_start_time
        epoch_start_time = time.time()
        # 初始化 iter_data_time
        iter_data_time = time.time()
        # 初始化 epoch_iter
        epoch_iter = 0

        for i, data in enumerate(dataset): # enumerate(dataset)每次都會讀入一個 batch 的資料
            
            # 初始化 iter_start_time
            iter_start_time = time.time()

            # 計算每次 iter 所花的時間（參考迴圈結尾 iter_data_time）
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            # 重置 visualizer
            visualizer.reset()
            
            # 加上每次 iter 讀的 batchSize (預設1)
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            
            # 建立 input real_A & real_B
            # it not only sets the input data with mask, but also sets the latent mask.
            model.set_input(data) 

            # 依照 display_freq 去 set_show_map_true()？ 
            # Additonal, should set it before 'optimize_parameters()'.
            if total_steps % opt.display_freq == 0: # default:100
                if opt.show_flow:
                    model.set_show_map_true()
            
            model.optimize_parameters()

            # 依照 display_freq 去 display_current_results
            if total_steps % opt.display_freq == 0: # default:100
                save_result = total_steps % opt.update_html_freq == 0
                if opt.show_flow:
                    model.set_flow_src()
                    model.set_show_map_false()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # 依照 print_freq 去 plot_current_losses
            if total_steps % opt.print_freq == 0: # default:50
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize # 平均每張圖的時間
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)
            
            # 依照 save_latest_freq 去 save_networks
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                        (epoch, total_steps))
                model.save_networks('latest')

            # 取得現在時間放入 iter_data_time
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
    