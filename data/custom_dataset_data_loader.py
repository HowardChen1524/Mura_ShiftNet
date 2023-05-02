#-*-coding:utf-8-*-
import torch.utils.data
from data.base_data_loader import BaseDataLoader
import random
import csv
import pandas as pd

def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned_resized':
        from data.aligned_dataset_resized import AlignedDatasetResized
        dataset = AlignedDatasetResized()

    elif opt.dataset_mode == 'aligned_sliding':
        from data.aligned_dataset_sliding import AlignedDatasetSliding
        dataset = AlignedDatasetSliding()

    elif opt.dataset_mode == 'sup_unsup_dataset':
        from data.sup_unsup_dataset import SupUnsupDataset
        dataset = SupUnsupDataset()
    
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