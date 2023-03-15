from data.custom_dataset_data_loader import CustomDatasetDataLoader
from collections import defaultdict
# CreateDataLoader -> CustomDatasetDataLoader 
# -> BaseDataLoader 
# -> CreateDataset
def CreateDataLoader(opt):
    if opt.isTrain:
        data_loader = CustomDatasetDataLoader()
        print(data_loader.name())
        data_loader.initialize(opt)
        return data_loader
    else:
        # for testing
        loaders = defaultdict()
        if opt.testing_normal_dataroot != '':
            n_data_loader = CustomDatasetDataLoader()
            print(n_data_loader.name())
            opt.dataroot = opt.testing_normal_dataroot
            n_data_loader.initialize(opt)
            loaders['normal'] = n_data_loader
        if opt.testing_smura_dataroot != '':
            s_data_loader = CustomDatasetDataLoader()
            print(s_data_loader.name())
            opt.dataroot = opt.testing_smura_dataroot
            s_data_loader.initialize(opt)
            loaders['smura'] = s_data_loader
        return loaders
