import math
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def get_custom_folds(train_df):
    train_df_all = train_df.copy()

    # count images per prediction_id
    train_df['prediction_id'] = train_df['patient_id'].astype(str).str.cat(train_df['laterality'], sep='_')
    train_df['prediction_id'] = train_df['patient_id'].astype(str).str.cat(train_df['laterality'], sep='_')
    num_count = train_df[['prediction_id', 'image_id']].groupby('prediction_id').count().reset_index()
    count_map = {pred_id: img_id for pred_id, img_id in zip(num_count['prediction_id'].values, num_count['image_id'].values)}
    train_df['count'] = train_df['prediction_id'].map(count_map)

    # group by patient_id and stratify by age, implant, machine_id, cancer, biopsy, BIRADS, density, and count
    train_df = train_df.groupby('prediction_id').first().reset_index()
    dummy = train_df[['patient_id', 'age', 'implant', 'machine_id']].groupby('patient_id').first()
    machine2int = {machine_id: n for n, machine_id in enumerate(train_df[['machine_id', 'cancer']].groupby('machine_id').mean().sort_values('cancer').index.values)}
    dummy['machine_id'] = dummy['machine_id'].apply(lambda x: machine2int[x])
    dummy2 = train_df[['patient_id', 'cancer', 'biopsy', 'count']].groupby('patient_id').mean()
    dummy3 = train_df[['patient_id', 'BIRADS']].groupby('patient_id').min().fillna(-1)
    dummy4 = train_df[['patient_id', 'density']].groupby('patient_id').max().fillna('E')
    dummy4['density'] = dummy4['density'].map({'E': -1, 'D': 0, 'C': 1, 'B': 2, 'A': 3})
    dummy = pd.concat([dummy, dummy2, dummy3, dummy4], axis=1)
    dummy['age'] = dummy['age'].fillna(dummy['age'].mean())
    dummy['fold'] = -1
    mskf = MultilabelStratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    for fold, (trn_ind, val_ind) in enumerate(mskf.split(dummy, dummy.values)):
        dummy.iloc[val_ind, -1] = fold
    dummy = dummy.reset_index()
    dummy['patient_id'] = dummy['patient_id'].astype('int')
    fold_map = {patient_id: fold for patient_id, fold in zip(dummy['patient_id'].values, dummy['fold'].values)}
    
    # show some stat regarding each fold
    train_df = train_df.merge(dummy[['patient_id','fold']], on='patient_id', how='left')
    for fold in range(config.n_folds):
        trn_ind = train_df[train_df['fold'] != fold].index
        val_ind = train_df[train_df['fold'] == fold].index
        print(f'=========== Fold {fold} ===========')
        print(f'Train {len(trn_ind)}', end='')
        _, counts = np.unique(train_df.loc[trn_ind, 'cancer'].values, return_counts=True)
        print(f'        (positive {counts[1]}, negative {counts[0]})')
        print(f'Validation {len(val_ind)}', end='')
        _, counts = np.unique(train_df.loc[val_ind, 'cancer'].values, return_counts=True)
        print(f'    (positive {counts[1]}, negative {counts[0]})')
    
    # concat 'fold' column on train_df_all
    train_df_all['fold'] = train_df_all['patient_id'].map(fold_map)
    return train_df_all

# https://www.kaggle.com/code/sohier/probabilistic-f-score
def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
            # cfp += 1 - prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return 0

def pfbeta_binarized(labels, predictions):
    positives = predictions[labels == 1]
    scores = []
    for th in positives:
        binarized = (predictions >= th).astype('int')
        score = pfbeta(labels, binarized, 1)
        scores.append(score)
    return np.max(scores)

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def pr_auc(y_true, y_pred):
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    score = auc(recall, precision)
    return score