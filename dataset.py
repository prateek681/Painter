from torchvision import transforms
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
#taken from https://www.kaggle.com/nachiket273/cyclegan-pytorch by @NACHIKET273
#changed a little for understandability
#creates dataset that feeds photo/monet noise, label
class ImageDataset(Dataset):
    def __init__(self, monet_dir, photo_dir, normalize=True):
        super().__init__()
        #folder with monets
        self.monet_dir = monet_dir
        #folder with photos
        self.photo_dir = photo_dir
        self.monet_idx = dict()
        self.photo_idx = dict()
        if normalize:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor()                               
            ])
        #iterate over all monets and store them in dict by index
        for i, monet in enumerate(os.listdir(self.monet_dir)):
            self.monet_idx[i] = monet
            
        #iterate over all photos and store them in dict by index
        for i, photo in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = photo

    def __getitem__(self, idx):
        rand_idx = int(np.random.uniform(0, len(self.monet_idx.keys())))
        photo_path = os.path.join(self.photo_dir, self.photo_idx[rand_idx])
        monet_path = os.path.join(self.monet_dir, self.monet_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        monet_img = Image.open(monet_path)
        monet_img = self.transform(monet_img)
        return photo_img, monet_img

    def __len__(self):
        return min(len(self.monet_idx.keys()), len(self.photo_idx.keys()))
    
    
class PhotoDataset(Dataset):
    def __init__(self, photo_dir, size=(256, 256), normalize=True):
        super().__init__()
        self.photo_dir = photo_dir
        self.photo_idx = dict()
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))                                
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(size),
                transforms.ToTensor()                               
            ])
        for i, fl in enumerate(os.listdir(self.photo_dir)):
            self.photo_idx[i] = fl

    def __getitem__(self, idx):
        photo_path = os.path.join(self.photo_dir, self.photo_idx[idx])
        photo_img = Image.open(photo_path)
        photo_img = self.transform(photo_img)
        return photo_img

    def __len__(self):
        return len(self.photo_idx.keys())