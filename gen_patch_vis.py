import numpy as np

from options.test_options import TestOptions

from data.data_loader import CreateDataLoader
from models import create_model
from util.utils_howard import mkdir, set_seed
                              
def initail_setting():
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle

    set_seed(2022)
    
    return opt, opt.gpu_ids[0]

def main(opt):
    model = create_model(opt)
    data_loader = CreateDataLoader(opt)

    dataset = data_loader['smura']
    fn_log = []
    model_pred_t_list = []
    combine_t_list = []
    denoise_t_list = []
    export_t_list = []
    
    for i, data in enumerate(dataset):
        if i >= opt.smura_how_many:
            break
        
        fn = data['A_paths'][0].split('/')[-1]
        print(f"Image num {i}: {fn}")
        fn_log.append(fn)

        # (1,mini-batch,c,h,w) -> (mini-batch,c,h,w)，會有多一個維度是因為 dataloader batchsize 設 1
        bs, ncrops, c, h, w = data['A'].size()
        data['A'] = data['A'].view(-1, c, h, w)

        bs, ncrops, c, h, w = data['B'].size()
        data['B'] = data['B'].view(-1, c, h, w)
        
        bs, ncrops, c, h, w = data['M'].size()
        data['M'] = data['M'].view(-1, c, h, w)
        
        # 建立 input real_A & real_B
        # it not only sets the input data with mask, but also sets the latent mask.
        model.set_input(data)
        t = model.visualize_diff(fn)
        model_pred_t_list.append(t[0])
        combine_t_list.append(t[1])
        denoise_t_list.append(t[2])
        export_t_list.append(t[3])

    print(f"model_pred time cost mean: {np.mean(model_pred_t_list)}")
    print(f"combine time cost mean: {np.mean(combine_t_list)}")
    print(f"denoise time cost mean: {np.mean(denoise_t_list)}")
    print(f"export time cost mean: {np.mean(export_t_list)}")

if __name__ == "__main__":
    opt, gpu = initail_setting()  
    main(opt)