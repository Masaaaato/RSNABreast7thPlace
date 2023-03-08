import sys
import os
import gc
import pickle
import time
import random
import math
import shutil
import argparse
from argparse import Namespace

import yaml
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import timm
from timm.scheduler import CosineLRScheduler
from albumentations  import *
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
print('torch version:', torch.__version__)
print('timm version:', timm.__version__)

# load custom function
sys.path.append('.')
from custom_utils import get_custom_folds, pfbeta, pfbeta_binarized, pr_auc, AverageMeter, asMinutes, timeSince
from dataset_utils import MammoDataset, MammoDataset_Lat, MammoDataset_Ext, MammoDataset_Ext_Lat

### load configuration from .yaml
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config path(.yaml)')
    args = parser.parse_args()
    return args
args = get_args()
config = yaml.safe_load(open(args.config_path, 'r'))
config = Namespace(**config)
INPUT_BASE = Path(config.INPUT_BASE)
OUTPUT_BASE = Path(config.OUTPUT_BASE)

device = config.DEVICE if torch.cuda.is_available() else 'cpu'
print('device:', device)

#### Model ####
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=False):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        ret = gem(x, p=self.p, eps=self.eps)
        return ret

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

class MammoModel(nn.Module):
    def __init__(self, name, *, pretrained=False, in_chans=1, p=3, p_trainable=False, eps=1e-6):
        super().__init__()
        model = timm.create_model(name, pretrained=pretrained, in_chans=in_chans)
        clsf = model.default_cfg['classifier']
        n_features = model._modules[clsf].in_features
        model._modules[clsf] = nn.Identity()
        
        self.fc = nn.Linear(n_features, 1) # ccancer
        self.model = model

        self.pool = nn.Sequential(
            GeM(p=p, eps=eps, p_trainable=p_trainable),
            nn.Flatten())
    
    def forward(self, x):
        # x = self.model(x)
        x = self.model.forward_features(x)
        x = self.pool(x)
        logits = self.fc(x)
        return logits

#### Dataset ####
class MammoDatasetTTA(Dataset):
    def __init__(self, df):
        self.df = df
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img_id = f"{data['image_id']}.png"
        path = str(INPUT_BASE.joinpath('external_data', str(data['patient_id']), img_id))
        # img = pickle.load(open(path, 'rb'))
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        img_h = HorizontalFlip(p=1)(image=img)['image']
        img_v = VerticalFlip(p=1)(image=img)['image']
        img_vh = HorizontalFlip(p=1)(image=img_v)['image']

        img = np.stack([img, img_h, img_v, img_vh], axis=0)
        # img = np.stack([img, img_v], axis=0)
        img = img.astype('float32')
        img -= img.min()
        img /= img.max()
        img = torch.tensor((img - config.MEAN)/config.STD, dtype=torch.float32)
        return img.unsqueeze(1)

#### Load Data ####
patient_ids = []
image_ids = []
for path in INPUT_BASE.joinpath('external_data').rglob('*.png'):
    parts = path.parts
    patient_ids.append(parts[-2])
    image_ids.append(parts[-1][:-4])
vindr_df = pd.DataFrame({'patient_id': patient_ids, 'image_id': image_ids})

# inference
for fold in range(config.n_folds):
    model = MammoModel(config.name, pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(OUTPUT_BASE.joinpath('models', f'efficientnetb5_seed_{config.seed}_fold{fold}_best_score_ver{config.VER}.pth'), map_location=device)['model'])
    model.eval()
    
    ds = MammoDatasetTTA(vindr_df)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True, drop_last=False)
    
    predictions = []
    for inputs in tqdm(dl):
        inputs = inputs.reshape(-1, 1, config.SIZE[0], config.SIZE[1])
        inputs = inputs.to(device)
        with torch.no_grad():
            preds = model(inputs)
        preds = preds.squeeze(1).sigmoid().to('cpu').numpy()
        preds = preds.reshape(-1, 4).mean(axis=1)
        predictions.append(preds)
    predictions = np.concatenate(predictions)
    vindr_df[f'fold{fold}_pred'] = predictions

vindr_anno = pd.read_csv(INPUT_BASE.joinpath('VinDrMammo_breast-level_annotations.csv'))
vindr_df = vindr_df.merge(vindr_anno[['image_id', 'laterality', 'view_position', 'breast_birads', 'breast_density']], on='image_id').reset_index(drop=True)
vindr_df['prediction_id'] = vindr_df['patient_id'].str.cat(vindr_df['laterality'], sep='_')
agg_preds = vindr_df[['prediction_id', 'fold0_pred', 'fold1_pred', 'fold2_pred', 'fold3_pred']].groupby('prediction_id').mean().reset_index()
vindr_df = vindr_df[['patient_id', 'image_id', 'laterality', 'view_position', 'breast_birads', 'breast_density', 'prediction_id']].merge(agg_preds, on='prediction_id').reset_index(drop=True)
vindr_df.to_csv(INPUT_BASE.joinpath('VinDr_pl_agg.csv'))
