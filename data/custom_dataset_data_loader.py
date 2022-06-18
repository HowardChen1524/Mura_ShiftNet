#-*-coding:utf-8-*-
import torch.utils.data
from data.base_data_loader import BaseDataLoader
import random

def CreateDataset(opt):
    dataset = None
    # 根據 param 選擇 dataset_mode
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()

    elif opt.dataset_mode == 'aligned_resized':
        from data.aligned_dataset_resized import AlignedDatasetResized
        dataset = AlignedDatasetResized()

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

        # 06/05 add random choose 10000 from train dataset
        if opt.isTrain:
            self.dataset = torch.utils.data.Subset(self.dataset,random.sample(list(range(len(self.dataset))), 10000))
            print("Success random choose!")
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches, # if true, 照圖片順序建立 batches, 反之 random
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
    
    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i*self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data