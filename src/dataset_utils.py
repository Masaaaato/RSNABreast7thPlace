from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import cv2

# Breast Level
class MammoDataset(Dataset):
    def __init__(self, df, config, train=True, tfms=None, windowing=False):
        self.df = df
        self.train = train
        self.tfms = tfms
        self.config = config
        self.INPUT_BASE = Path(config.INPUT_BASE)
        self.windowing = windowing
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_id = f"{data['image_id']}.png"
        path = str(self.INPUT_BASE.joinpath("train_images", str(data['patient_id']), img_id))
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.tfms:
            augmented = self.tfms(image=img)
            img = augmented['image']
        img = img.astype('float32')

        if self.windowing:
            # sigmoid windowing
            img /= 255
            img = img * (data['max'] - data['min']) + data['min']
            if data['rev'] == 1:
                img = data['rev_max'] - img
            img = data['y_range'] / (1 + np.exp(-4 * (img - data['center']) / data['width']))
            if data['rev'] == 1:
                img = np.amax(img) - img

        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - self.config.MEAN)/self.config.STD, dtype=torch.float32)
        if self.train:
            return img.unsqueeze(0), torch.tensor(data['cancer'], dtype=torch.long)
        else:
            return img.unsqueeze(0)

# Laterality Level
class MammoDataset_Lat(Dataset):
    def __init__(self, df, config, train=True, tfms=None, size=1520, windowing=False):
        self.df = df
        self.train = train
        self.tfms = tfms
        self.config = config
        self.INPUT_BASE = Path(config.INPUT_BASE)
        self.size = size
        self.windowing = windowing
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        imgs = []
        for img_id in data['image_id']:
            img_id = f"{img_id}.png"
            path = str(self.INPUT_BASE.joinpath("train_images", str(data['patient_id']), img_id))
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)

        if self.windowing:
            # sigmoid windowing
            imgs = imgs.astype('float32')
            imgs /= 255
            imgs = imgs * (data['max'] - data['min']) + data['min']
            if data['rev'] == 1:
                imgs = data['rev_max'] - imgs
            imgs = data['y_range'] / (1 + np.exp(-4 * (imgs - data['center']) / data['width']))
            if data['rev'] == 1:
                imgs = np.amax(imgs) - imgs
            imgs -= imgs.min()
            imgs /= imgs.max()
            imgs *= 255

        imgs = cv2.resize(imgs, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.tfms:
            augmented = self.tfms(image=imgs)
            imgs = augmented['image']
        imgs = imgs.astype('float32')
        imgs -= imgs.min()
        imgs /= imgs.max()
        imgs = torch.tensor((img - self.config.MEAN)/self.config.STD, dtype=torch.float32)
        if self.train:
            return imgs.unsqueeze(0), torch.tensor(data['cancer'], dtype=torch.long)
        else:
            return imgs.unsqueeze(0)

# Breast Level, with external data
class MammoDataset_Ext(Dataset):
    def __init__(self, df, config, train=True, tfms=None, ext_df=None, windowing=False):
        self.df = df
        self.train = train
        self.tfms = tfms
        self.ext_df = ext_df
        self.config = config
        self.INPUT_BASE = Path(config.INPUT_BASE)
        self.windowing = windowing
        
    def __len__(self):
        if self.train:
            return len(self.df)+len(self.ext_df)
        else:
            return len(self.df)
    
    def __getitem__(self, idx):
        if idx < len(self.df):
            data = self.df.iloc[idx]
            img_id = f"{data['image_id']}.png"
            path = str(self.INPUT_BASE.joinpath("train_images", str(data['patient_id']), img_id))
        else:
            idx -= len(self.df)
            data = self.ext_df.iloc[idx]
            img_id = f"{data['image_id']}.png"
            path = str(self.INPUT_BASE.joinpath("external_data", str(data['patient_id']), img_id))
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.tfms:
            augmented = self.tfms(image=img)
            img = augmented['image']
        img = img.astype('float32')

        if self.windowing:
            # sigmoid windowing
            img /= 255
            img = img * (data['max'] - data['min']) + data['min']
            if data['rev'] == 1:
                img = data['rev_max'] - img
            img = data['y_range'] / (1 + np.exp(-4 * (img - data['center']) / data['width']))
            if data['rev'] == 1:
                img = np.amax(img) - img
        img -= img.min()
        
        if img.max() == 0:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = img.astype('float32')
            if self.windowing:
                # sigmoid windowing
                img /= 255
                img = img * (data['max'] - data['min']) + data['min']
                if data['rev'] == 1:
                    img = data['rev_max'] - img
                img = data['y_range'] / (1 + np.exp(-4 * (img - data['center']) / data['width']))
                if data['rev'] == 1:
                    img = np.amax(img) - img
            img -= img.min()
            
        img /= img.max()
        img = torch.tensor((img - self.config.MEAN)/self.config.STD, dtype=torch.float32)
        return img.unsqueeze(0), torch.tensor(data['cancer'], dtype=torch.float32)

# Laterality Level, with external data
class MammoDataset_Ext_Lat(Dataset):
    def __init__(self, df, config, train=True, tfms=None, size=1520, ext_df=None, windowing=False):
        self.df = df
        self.train = train
        self.tfms = tfms
        self.size = (size, size)
        self.ext_df = ext_df
        self.config = config
        self.INPUT_BASE = Path(config.INPUT_BASE)
        self.windowing = windowing
        
    def __len__(self):
        if self.train:
            return len(self.df)+len(self.ext_df)
        else:
            return len(self.df)
    
    def __getitem__(self, idx):
        imgs = []
        if idx < len(self.df):
            data = self.df.iloc[idx]
            for img_id in data['image_id']:
                img_id = f"{img_id}.png"
                path = str(self.INPUT_BASE.joinpath("train_images", str(data['patient_id']), img_id))
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                imgs.append(img)
        else:
            idx -= len(self.df)
            data = self.ext_df.iloc[idx]
            for img_id in data['image_id']:
                img_id = f"{img_id}.png"
                path = str(self.INPUT_BASE.joinpath("external_data", str(data['patient_id']), img_id))
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                imgs.append(img)
        imgs = np.concatenate(imgs, axis=1)

        if self.windowing:
            # sigmoid windowing
            imgs = imgs.astype('float32')
            imgs /= 255
            imgs = imgs * (data['max'] - data['min']) + data['min']
            if data['rev'] == 1:
                imgs = data['rev_max'] - imgs
            imgs = data['y_range'] / (1 + np.exp(-4 * (imgs - data['center']) / data['width']))
            if data['rev'] == 1:
                imgs = np.amax(imgs) - imgs
            imgs -= imgs.min()
            imgs /= imgs.max()
            imgs *= 255

        imgs = cv2.resize(imgs, self.size, interpolation=cv2.INTER_AREA)
        if self.tfms:
            augmented = self.tfms(image=imgs)
            imgs = augmented['image']
        imgs = imgs.astype('float32')
        imgs -= imgs.min()
        imgs /= imgs.max()
        imgs = torch.tensor((imgs - self.config.MEAN)/self.config.STD, dtype=torch.float32)
        return imgs.unsqueeze(0), torch.tensor(data['cancer'], dtype=torch.float32)