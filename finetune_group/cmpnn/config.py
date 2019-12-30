# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:50:08 2019

@author: SY
"""
import pandas as pd


import os
cuda = True# False

gpu = 5
dataset_type = 'classification' # regression classification
data_path = f'../train.csv'
data_name = data_path.split("/")[-1].split(".")[0]
n_repeats = 1
epochs = 10# 15
batch_size = 32
split_type = 'crossval'
verbose = True


num_folds = 5
seed = 2050
depth = 3

init_lr = 1e-4
max_lr = 1e-3
final_lr = 1e-4
hidden_size = 300
dropout = 0
bias = True
ffn_num_layers = 2
ffn_hidden_size = 300
ensemble_size = 1
log_frequency = 10
warmup_epochs = 2
num_lrs = 1


# =============================================================================
# 
# =============================================================================
from cmpnn.data.feature import get_atom_bond_dim
atom_dim, bond_dim = get_atom_bond_dim()

save_dir = f'./ckpt_{data_name}'
checkpoint_dir = save_dir
checkpoint_paths = []
for root, _, files in os.walk(checkpoint_dir):
    for fname in files:
        if fname.endswith('.pt'):
            checkpoint_paths.append(os.path.join(root, fname))

import pandas as pd
df = pd.read_csv(data_path, nrows=1)
task_names = df.columns.tolist()[1:-1]
num_tasks = 1# len(task_names)

assert dataset_type in ['classification', 'regression']
if dataset_type == 'classification':
    metric = 'auc'
    minimize_score = False
else:
    metric = 'rmse'
    minimize_score = True