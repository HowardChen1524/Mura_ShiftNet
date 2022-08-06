
# CreateDataLoader -> CustomDatasetDataLoader 
# -> BaseDataLoader 
# -> CreateDataset
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader

    if opt.isTrain:
        # if nowValidation:
        data_loader = CustomDatasetDataLoader()
        print(data_loader.name())
        data_loader.initialize(opt)
        return data_loader
        # else:
        #     loaders = []
        #     n_data_loader = CustomDatasetDataLoader()
        #     print(n_data_loader.name())
        #     opt.dataroot = opt.validate_normal_dataroot
        #     n_data_loader.initialize(opt)
        #     loaders.append(n_data_loader)

        #     s_data_loader = CustomDatasetDataLoader()
        #     print(s_data_loader.name())
        #     opt.dataroot = opt.validate_smura_dataroot
        #     s_data_loader.initialize(opt)
        #     loaders.append(s_data_loader)
        #     return loaders
    else:
        if opt.dataset_mode == 'aligned_combined': # for type-c wei
            data_loader = CustomDatasetDataLoader()
            print(data_loader.name())
            data_loader.initialize(opt)
            return data_loader
        else: # change to defaultdict
            loaders = []
            n_data_loader = CustomDatasetDataLoader()
            print(n_data_loader.name())
            opt.dataroot = opt.testing_normal_dataroot
            n_data_loader.initialize(opt)
            loaders.append(n_data_loader)

            s_data_loader = CustomDatasetDataLoader()
            print(s_data_loader.name())
            opt.dataroot = opt.testing_smura_dataroot
            s_data_loader.initialize(opt)
            loaders.append(s_data_loader)
            
        return loaders
