
from torch.utils.data import Dataset
from PIL import Image
class AI9_Dataset(Dataset):
    def __init__(self, feature, target, name, transform=None):
        self.X = feature # path
        self.Y = target # label
        self.N = name # name
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = Image.open(self.X[idx])
        
        return self.transform(img), self.Y[idx], self.N[idx]
