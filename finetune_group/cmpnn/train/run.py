# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 00:09:49 2019

@author: SY
"""

import os
import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR

from cmpnn.train.utils import evaluate, evaluate_predictions, predict, train, get_acc
from cmpnn.data import StandardScaler
from cmpnn.model import build_model
from cmpnn.nn_utils import param_count
from cmpnn.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint,\
    makedirs, save_checkpoint


from cmpnn import config
from cmpnn.data.utils import load_data

def run(df_train, df_val, df_test, model_id=0):
    try:
        torch.cuda.set_device(config.gpu)
    except:
        print('using cpu')
    # Get data
    if config.verbose:
        print('Loading data')
    train_data = load_data(df_train)
    val_data = load_data(df_val)
    test_data = load_data(df_test)
    
    config.num_tasks = 1# train_data.num_tasks()
    if config.verbose:
        print(f'Number of tasks = {config.num_tasks}')

    features_scaler = None

    config.train_data_size = len(train_data)
    
    if config.verbose:
        print(f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if config.dataset_type == 'regression':
        if config.verbose:
            print('Fitting scaler')
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(config)
    metric_func = get_metric_func(metric=config.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    sum_test_preds = np.zeros((len(test_smiles), config.num_tasks))
    if config.dataset_type == 'classification':
        sum_test_acc = np.zeros((len(test_smiles), config.num_tasks))
        sum_test_prec = np.zeros((len(test_smiles), config.num_tasks))
        
    # Train ensemble of models
    for model_idx in range(config.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(config.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        # Load/build model
        if config.verbose:
            print(f'Building model {model_idx}')
        model = load_checkpoint(f'../train/ckpt_logp_caco_new200_clean/model_0/model_0.pt')

        if config.verbose:
            print(model)
            print(f'Number of parameters = {param_count(model):,}')
        if config.cuda:
            print('Moving model to cuda')
            model = model.cuda()
            print('done.')

        save_checkpoint(os.path.join(save_dir, f'model_{model_id}.pt'), model, scaler, features_scaler)

        best_score = float('inf') if config.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        optimizer = build_optimizer(model, config)
        scheduler = build_lr_scheduler(optimizer, config)
        for epoch in range(config.epochs):
            print(f'Epoch {epoch}')
            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                n_iter=n_iter,
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=config.num_tasks,
                metric_func=metric_func,
                batch_size=config.batch_size,
                dataset_type=config.dataset_type,
                scaler=scaler,
            )
            if config.dataset_type == 'regression':
                print(f'R2: {val_scores[1]:.4f}')
            else:
                acc_scores = val_scores[1]
                print(f'acc: {acc_scores:.4f}')
                avg_val_acc_score = np.nanmean(acc_scores)
            val_scores = val_scores[0]
            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            print(f'Validation {config.metric} = {avg_val_score:.6f}')
            if config.dataset_type == 'classification':
                print(f'Validation acc = {avg_val_acc_score:.6f}')
            # Save model checkpoint if improved validation score
            if config.minimize_score and avg_val_score < best_score or \
                    not config.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                if config.dataset_type == 'classification':
                    best_acc = avg_val_acc_score
                save_checkpoint(os.path.join(save_dir, f'model_{model_id}.pt'), model, scaler, features_scaler)        
                break

        # Evaluate on test set using model with best validation score
        print(f'Model {model_idx} best validation {config.metric} = {best_score:.6f} on epoch {best_epoch}')
        if config.dataset_type == 'classification':
            print(f'Model {model_idx} best validation acc = {best_acc:.6f}')
        model = load_checkpoint(os.path.join(save_dir, f'model_{model_id}.pt'))
        
        test_preds = predict(
            model=model,
            data=test_data,
            batch_size=config.batch_size,
            scaler=scaler
        )
        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=config.num_tasks,
            metric_func=metric_func,
            dataset_type=config.dataset_type,
        )
        if config.dataset_type == 'classification':
            test_acc, test_prec = get_acc(test_targets, test_preds)
            sum_test_prec += np.array(test_prec)
            sum_test_acc += np.array(test_acc)
        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        # Average test score
        avg_test_score = np.nanmean(test_scores)
        if config.verbose:
            print(f'Model {model_idx} test {config.metric} = {avg_test_score:.6f}')
        if config.dataset_type == 'classification':
            print(f'Model {model_idx} test acc = {np.nanmean(test_acc):.6f}')

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / config.ensemble_size).tolist()
    
    ensemble_scores = evaluate_predictions(
        preds=avg_test_preds,
        targets=test_targets,
        num_tasks=config.num_tasks,
        metric_func=metric_func,
        dataset_type=config.dataset_type,
    )
    if config.dataset_type == 'classification':
        ensemble_acc, ensemble_prec = get_acc(test_targets, avg_test_preds)
        

    # Average ensemble score
    avg_ensemble_test_score = np.nanmean(ensemble_scores)
    if config.dataset_type == 'classification':
        avg_ensemble_test_acc = np.nanmean(ensemble_acc)
        avg_ensemble_test_prec = np.nanmean(ensemble_prec)
        
    if config.verbose:
        print(f'Ensemble test {config.metric} = {avg_ensemble_test_score:.6f}')
        if config.dataset_type == 'classification':
            print(f'Ensemble test acc = {avg_ensemble_test_acc:.6f}')
            print(f'Ensemble test precision = {avg_ensemble_test_prec:.6f}')
            
    if config.dataset_type == 'classification':
        return ([avg_ensemble_test_score, avg_ensemble_test_acc, avg_ensemble_test_prec], avg_test_preds)
    return ([avg_ensemble_test_score], avg_test_preds)
