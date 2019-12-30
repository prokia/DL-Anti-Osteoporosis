# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:10:37 2019

@author: SY

"""
from cmpnn.data import MoleculeDataset, StandardScaler
import torch
import torch.nn as nn
import logging
from typing import Callable, List, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange
from cmpnn.nn_utils import NoamLR
from cmpnn import config
from sklearn import metrics

def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          n_iter: int = 0,
          logger: logging.Logger = None) -> int:
    
    model.train()
    data.shuffle()
    loss_sum, iter_count = 0, 0
    num_iters = len(data) // config.batch_size * config.batch_size  # don't use the last batch if it's small, for stability
    iter_size = config.batch_size
    
    if config.verbose:
        generater = trange
    else:
        generater = range
    
    for i in generater(0, num_iters, iter_size):
        # Prepare batch
        if i + config.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + config.batch_size])
        smiles_batch, target_batch = mol_batch.smiles(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        class_weights = torch.ones(targets.shape)

        if config.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds = model(batch)

        if config.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(mol_batch)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(mol_batch)

    return n_iter

def get_acc(y, pred):
    from sklearn.metrics import accuracy_score, precision_score
    import numpy as np
    pred = np.array(pred)
    best = -1
    T = -1
    for i in range(100):
        tmp = accuracy_score(y, np.where(pred>i/100, 1, 0))
        if tmp > best:
            best = tmp
            T = i
    prec = precision_score(y, np.where(pred>T/100, 1, 0))
    return best, prec

def evaluate(model: nn.Module,
             data: MoleculeDataset,
             num_tasks: int,
             metric_func: Callable,
             batch_size: int,
             dataset_type: str,
             scaler: StandardScaler = None) -> List[float]:
    preds = predict(
        model=model,
        data=data,
        batch_size=batch_size,
        scaler=scaler
    )

    targets = data.targets()

    results = evaluate_predictions(
        preds=preds,
        targets=targets,
        num_tasks=num_tasks,
        metric_func=metric_func,
        dataset_type=dataset_type,
    )
    if config.metric == 'rmse':
        r2 = metrics.r2_score(targets, preds)
        return results, r2
    acc, prec = get_acc(targets, preds)
    return results, acc


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    model.eval()

    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch = mol_batch.smiles()

        batch = smiles_batch
        with torch.no_grad():
            batch_preds = model(batch)

        batch_preds = batch_preds.data.cpu().numpy()

        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds


def evaluate_predictions(preds: List[List[float]],
                         targets: List[List[float]],
                         num_tasks: int,
                         metric_func: Callable,
                         dataset_type: str) -> List[float]:

    if len(preds) == 0:
        return [float('nan')] * num_tasks

    valid_preds = [[] for _ in range(num_tasks)]
    valid_targets = [[] for _ in range(num_tasks)]
    for i in range(num_tasks):
        for j in range(len(preds)):
            if targets[j][i] is not None:  # Skip those without targets
                valid_preds[i].append(preds[j][i])
                valid_targets[i].append(targets[j][i])

    results = []
    for i in range(num_tasks):
        if dataset_type == 'classification':
            nan = False
            if all(target == 0 for target in valid_targets[i]) or all(target == 1 for target in valid_targets[i]):
                nan = True
                print('Warning: Found a task with targets all 0s or all 1s')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                print('Warning: Found a task with predictions all 0s or all 1s')

            if nan:
                results.append(float('nan'))
                continue

        if len(valid_targets[i]) == 0:
            continue

        if dataset_type == 'multiclass':
            results.append(metric_func(valid_targets[i], valid_preds[i], labels=list(range(len(valid_preds[i][0])))))
        else:
            results.append(metric_func(valid_targets[i], valid_preds[i]))

    return results


