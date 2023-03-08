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
from .custom_utils import get_custom_folds, pfbeta, pfbeta_binarized, pr_auc, AverageMeter, asMinutes, timeSince
from .dataset_utils import MammoDataset, MammoDataset_Lat, MammoDataset_Ext, MammoDataset_Ext_Lat

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
model_base_name = 'efficientv2_s' if 'efficientnetv2' in config.name else 'efficientnetb5'

device = config.DEVICE if torch.cuda.is_available() else 'cpu'
print('device:', device)

### Setting
def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
seed_everything(config.seed)

def get_aug(p=1.0, a=10, s=10):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        # RandomRotate90(),
#         ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.8, 
#                          border_mode=cv2.BORDER_REFLECT)
        # OneOf([Affine(rotate=20, translate_percent=0.1, scale=[0.8,1.2], shear=20)])
        Affine(rotate=20, translate_percent=0.1, scale=[0.8,1.2], shear=20),
        ElasticTransform(alpha=a, sigma=s)
    ], p=p)

### Model
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
        
        self.fc = nn.Linear(n_features, 1) # cancer
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

### Train Utils
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=config.apex)
    losses = AverageMeter()
    start = end = time.time()
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=config.apex):
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # batch scheduler
        # scheduler.step()
        end = time.time()
        if step % config.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}'
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(device)
        labels = labels.float().to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
        loss = criterion(y_preds.view(-1, 1), labels.view(-1, 1))
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.squeeze(1).sigmoid().to('cpu').numpy())
        end = time.time()
        if step % config.print_freq == 0 or step == (len(valid_loader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def train_loop(folds, fold):
    print(f'================== fold: {fold} training ======================')
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    
    if config.external:
        ext_fold = vindr_df[['patient_id', 'image_id', f'fold{fold}_pred']].reset_index(drop=True)
        ext_fold.columns = ['patient_id', 'image_id', 'cancer']
        if config.level == 'breast':
            train_dataset = MammoDataset_Ext(train_folds, config, tfms=get_aug(a=config.a, s=config.s), ext_df=ext_fold, windowing=config.windowing)
            valid_dataset = MammoDataset_Ext(valid_folds, config, train=False, windowing=config.windowing)
        else:
            train_dataset = MammoDataset_Ext_Lat(train_folds, config, tfms=get_aug(a=config.a, s=config.s), ext_df=ext_fold, windowing=config.windowing)
            valid_dataset = MammoDataset_Ext_Lat(valid_folds, config, train=False, windowing=config.windowing)
    else:
        if config.level == 'breast':
            train_dataset = MammoDataset(train_folds, config, tfms=get_aug(a=config.a, s=config.s), windowing=config.windowing)
            valid_dataset = MammoDataset(valid_folds, config)
        else:
            train_dataset = MammoDataset_Lat(train_folds, config, tfms=get_aug(a=config.a, s=config.s), windowing=config.windowing)
            valid_dataset = MammoDataset_Lat(valid_folds, config)
    
    train_loader = DataLoader(train_dataset,
                             batch_size=config.batch_size,
                             shuffle=True,
                             num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                             batch_size=config.batch_size,
                             shuffle=False,
                             num_workers=4, pin_memory=True, drop_last=False)
    
    model = MammoModel(config.name, pretrained=True)
    model.to(device)
    
    optimizer = Adam(model.parameters(), lr=config.lr)
    scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=config.epochs_warmup, num_training_steps=config.epochs, num_cycles=config.num_cycles
            )

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    
    best_score = 0.
    best_aucroc = 0.
    best_prauc = 0.
    for epoch in range(config.epochs):
        start_time = time.time()
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, device)
        scheduler.step()
        # eval
        avg_val_loss, predictions = valid_fn(valid_loader, model, criterion, device)
        valid_folds['prediction'] = predictions
        valid_agg = valid_folds[['patient_id', 'laterality', 'cancer', 'prediction', 'fold']].groupby(['patient_id', 'laterality']).mean()
        score = pfbeta_binarized(valid_agg['cancer'].values, valid_agg['prediction'].values)
        prauc = pr_auc(valid_agg['cancer'].values, valid_agg['prediction'].values)
        aucroc = roc_auc_score(valid_agg['cancer'].values, valid_agg['prediction'].values)
        
        elapsed = time.time() - start_time
        
        print(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        print(f'Epoch {epoch+1} - pF Score: {score:.4f}, PR-AUC Score: {prauc:.4f}, AUC-ROC Score: {aucroc:.4f}')
        
        if best_prauc < prauc:
            best_prauc = prauc
            # torch.save({'model': model.state_dict(),
            #             'predictions': predictions},
            #             OUTPUT_BASE.joinpath("models", f"{model_base_name}_seed_{config.seed}_fold{fold}_best_prauc_ver{config.VER}.pth"))
            
        if best_aucroc < aucroc:
            best_aucroc = aucroc
            # torch.save({'model': model.state_dict(),
            #             'predictions': predictions},
            #             OUTPUT_BASE.joinpath("models", f"{model_base_name}_seed_{config.seed}_fold{fold}_best_aucroc_ver{config.VER}.pth"))
            
        if best_score < score:
            best_score = score
            print(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                        OUTPUT_BASE.joinpath("models", f"{model_base_name}_seed_{config.seed}_fold{fold}_best_score_ver{config.VER}.pth"))
        
    predictions = torch.load(OUTPUT_BASE.joinpath("models", f'{model_base_name}_seed_{config.seed}_fold{fold}_best_score_ver{config.VER}.pth'), map_location='cpu')['predictions']
    valid_folds['prediction'] = predictions
    print(f'[Fold{fold}] Best pF Score: {best_score}, PR-AUC Score: {best_prauc}, AUC-ROC Score: {best_aucroc:.4f}')
    torch.cuda.empty_cache()
    gc.collect()
    return valid_folds

#################
##### Train #####
#################

### Load Train
train_df = pd.read_csv(INPUT_BASE.joinpath('train.csv'))
# cv splitting, grouped by patient_id and stratified by age, implant, machine_id, cancer, biopsy, BIRADS, density, and the num of images
# get_custom_folds returns train_df with 'fold' column
train_df = get_custom_folds(train_df)

if config.external:
    vindr_df = pd.read_csv(INPUT_BASE.joinpath('VinDr_pl_agg.csv'))

if config.level == 'laterality':
    train_df = train_df.loc[[v in ['CC', 'MLO'] for v in train_df['view'].values]].reset_index(drop=True)
    image_ids = train_df.groupby(['patient_id', 'laterality'])['image_id'].apply(list).reset_index()['image_id'].values
    train_df = train_df.groupby(['patient_id', 'laterality']).first().reset_index()
    train_df['image_id'] = image_ids
    if config.external:
        image_ids = vindr_df.groupby(['patient_id', 'laterality'])['image_id'].apply(list).reset_index()['image_id'].values
        vindr_df = vindr_df.groupby(['patient_id', 'laterality']).first().reset_index()
        vindr_df['image_id'] = image_ids

oof_df = pd.DataFrame()
for fold in range(config.n_folds):
    seed_everything(config.seed)
    _oof_df = train_loop(train_df, fold)
    oof_df = pd.concat([oof_df, _oof_df])
oof_df = oof_df.reset_index(drop=True)
oof_df_agg = oof_df[['patient_id', 'laterality', 'cancer', 'prediction', 'fold']].groupby(['patient_id', 'laterality']).mean()
print('================ CV ================')
score = pfbeta_binarized(oof_df_agg['cancer'].values, oof_df_agg['prediction'].values)
prauc = pr_auc(oof_df_agg['cancer'].values, oof_df_agg['prediction'].values)
aucroc = roc_auc_score(oof_df_agg['cancer'].values, oof_df_agg['prediction'].values)
print(f'Score: {score}, PR-AUC: {prauc}, AUC-ROC: {aucroc}')
oof_df.to_pickle(OUTPUT_BASE.joinpath('preds', f'oof_df_ver{config.VER}_seed{config.seed}.pkl'))

