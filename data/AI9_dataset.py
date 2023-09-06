from torch.utils.data import Dataset
from PIL import Image
import tensorflow as tf
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode

class tjwei_augumentation(object):
  def __call__(self, img):
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.rgb_to_grayscale(img)
    img2 = tf.image.sobel_edges(img[None, ...])
    img = tf.concat([img, img2[0, :, :, 0]], 2)
    image_array = tf.keras.preprocessing.image.array_to_img(img)
    
    return image_array
  def __repr__(self):
    return self.__class__.__name__+'()'

data_transforms = {
  "train": transforms.Compose([
      # transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
      # transforms.RandomHorizontalFlip(),
      # transforms.RandomVerticalFlip(),
      # tjwei_augumentation(),
      # transforms.ToTensor(),
  ]),
  "test": transforms.Compose([
      transforms.Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
      tjwei_augumentation(),
      transforms.ToTensor()
  ])
}

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
