
# CreateDataLoader -> CustomDatasetDataLoader 
# -> BaseDataLoader 
# -> CreateDataset
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader

    if opt.isTrain:
        data_loader = CustomDatasetDataLoader()
        print(data_loader.name())
        data_loader.initialize(opt)
        return data_loader
    else:
        n_data_loader = CustomDatasetDataLoader()
        print(n_data_loader.name())
        opt.dataroot = opt.testing_normal_dataroot
        n_data_loader.initialize(opt)

        s_data_loader = CustomDatasetDataLoader()
        print(s_data_loader.name())
        opt.dataroot = opt.testing_smura_dataroot
        s_data_loader.initialize(opt)
        return [n_data_loader, s_data_loader]
