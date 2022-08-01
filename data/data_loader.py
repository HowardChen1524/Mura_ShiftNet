
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
        if opt.dataset_mode == 'aligned_combined': # for type-c wei
            data_loader = CustomDatasetDataLoader()
            print(data_loader.name())
            data_loader.initialize(opt)
            return data_loader
        else: # change to defaultdict
            loaders = []
            if opt.normal_how_many != 0:
                n_data_loader = CustomDatasetDataLoader()
                print(n_data_loader.name())
                opt.dataroot = opt.testing_normal_dataroot
                n_data_loader.initialize(opt)
                loaders.append(n_data_loader)
            else:
                print("=====normal_how_many empty!=====")

            if opt.smura_how_many != 0:
                s_data_loader = CustomDatasetDataLoader()
                print(s_data_loader.name())
                opt.dataroot = opt.testing_smura_dataroot
                s_data_loader.initialize(opt)
                loaders.append(s_data_loader)
            else:
                print("=====smura_how_many empty!=====")
            
        return loaders
