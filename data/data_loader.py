
def CreateDataLoader(opt, isTrain):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())

    data_loader.initialize(opt, isTrain)
    return data_loader
