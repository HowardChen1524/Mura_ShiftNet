#-*-coding:utf-8-*-
import torch.utils.data
from data.base_data_loader import BaseDataLoader
import random
import csv
import pandas as pd

def CreateDataset(opt):
    dataset = None
    # 根據 param 選擇 dataset_mode
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()

    elif opt.dataset_mode == 'aligned_resized':
        from data.aligned_dataset_resized import AlignedDatasetResized
        dataset = AlignedDatasetResized()

    # 06/28 add sliding
    elif opt.dataset_mode == 'aligned_sliding':
        from data.aligned_dataset_sliding import AlignedDatasetSliding
        dataset = AlignedDatasetSliding()

     # 07/30 add combined testing normal & smura
    elif opt.dataset_mode == 'aligned_combined':
        from data.aligned_dataset_combined import AlignedDatasetCombined
        dataset = AlignedDatasetCombined()

    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()

    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        
        # # 06/05 add random choose from train dataset
        # if self.opt.isTrain:
        #     if not self.opt.continue_train: # if no continue train, save the random choosed filename           
        #         self.random = random.sample(list(range(len(self.dataset))), self.opt.random_choose_num)
        #         self.dataset = torch.utils.data.Subset(self.dataset, self.random)
        #         recover_list = []
        #         for i, data in enumerate(self.dataset):
        #             print(i)
        #             recover_list.append(data['A_paths'][len(self.opt.dataroot):].replace('.png','.bmp'))
        #         recover_df = pd.DataFrame(recover_list, columns=['PIC_ID'])
        #         recover_df.to_csv('./training_imgs.csv', index=False, columns=['PIC_ID'])
        #         print("Success random choose!")
        #     else:
        #         print("Start continue train!")

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.opt.batchSize,
            shuffle=not self.opt.serial_batches, # if true, create batches in order, otherwise random, self.opt.serial_batches default false
            num_workers=int(self.opt.nThreads))
        

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i*self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data