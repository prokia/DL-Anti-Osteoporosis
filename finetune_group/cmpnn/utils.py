# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:11:00 2019

@author: SY
"""
import math
import os
from typing import Callable, List, Tuple, Union
from argparse import Namespace

from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from cmpnn.data import StandardScaler
from cmpnn.model import build_model
from cmpnn.nn_utils import NoamLR
from cmpnn import config


def makedirs(path: str, isfile: bool = False):
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str,
                    model,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None):
    state = {
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    torch.save(state, path)


def load_checkpoint(path: str):
    # Load model and config
    state = torch.load(path, map_location=lambda storage, loc: storage)
    loaded_state_dict = state['state_dict']



    # Build model
    model = build_model()
    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            print(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            # print(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if config.cuda:
        print('Moving model to cuda')
        model = model.cuda()

    return model


def load_scalers(path: str) -> Tuple[StandardScaler, StandardScaler]:
    state = torch.load(path, map_location=lambda storage, loc: storage)

    scaler = StandardScaler(state['data_scaler']['means'],
                            state['data_scaler']['stds']) if state['data_scaler'] is not None else None
    features_scaler = StandardScaler(state['features_scaler']['means'],
                                     state['features_scaler']['stds'],
                                     replace_nan_token=0) if state['features_scaler'] is not None else None

    return scaler, features_scaler


def load_config(path: str) -> Namespace:
    return torch.load(path, map_location=lambda storage, loc: storage)['config']


def load_task_names(path: str) -> List[str]:
    return load_config(path).task_names


def get_loss_func(config):
    if config.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    if config.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')
    
    if config.dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')

    raise ValueError(f'Dataset type "{config.dataset_type}" not supported.')


def prc_auc(targets: List[int], preds: List[float]) -> float:
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)


def rmse(targets: List[float], preds: List[float]) -> float:
    return math.sqrt(mean_squared_error(targets, preds))


def mse(targets: List[float], preds: List[float]) -> float:
    return mean_squared_error(targets, preds)


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    if type(preds[0]) == list: # multiclass
        hard_preds = [p.index(max(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds] # binary prediction
    return accuracy_score(targets, hard_preds)


def get_metric_func(metric: str) -> Callable[[Union[List[int], List[float]], List[float]], float]:
    if metric == 'auc':
        return roc_auc_score

    if metric == 'prc-auc':
        return prc_auc

    if metric == 'rmse':
        return rmse
    
    if metric =='mse':
        return mse

    if metric == 'mae':
        return mean_absolute_error

    if metric == 'r2':
        return r2_score
    
    if metric == 'accuracy':
        return accuracy
    
    if metric == 'cross_entropy':
        return log_loss

    raise ValueError(f'Metric "{metric}" not supported.')


def build_optimizer(model: nn.Module, config) -> Optimizer:
    params = [{'params': model.ffn.parameters(), 'lr': config.init_lr, 'weight_decay': 0}]
    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, config, total_epochs: List[int] = None) -> _LRScheduler:
    # Learning rate scheduler
    return NoamLR(
        optimizer=optimizer,
        warmup_epochs=[config.warmup_epochs],
        total_epochs=total_epochs or [config.epochs] * config.num_lrs,
        steps_per_epoch=config.train_data_size // config.batch_size,
        init_lr=[config.init_lr],
        max_lr=[config.max_lr],
        final_lr=[config.final_lr]
    )


