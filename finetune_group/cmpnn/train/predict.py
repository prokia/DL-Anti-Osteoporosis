import numpy as np
import torch
from tqdm import tqdm

from cmpnn.train.utils import predict
from cmpnn.data import MoleculeDataset
from cmpnn.utils import load_checkpoint
from cmpnn import config
from cmpnn.data.utils import load_data

def prediction(df):
    try:
        torch.cuda.set_device(config.gpu)
    except:
        print('no gpu')
    num_tasks = 1# df.shape[1] - 1
    print('Loading data')
    test_data = load_data(df)
    print('Validating SMILES')
    valid_indices = [i for i in range(len(test_data)) if test_data[i].mol is not None]
    full_data = test_data
    test_data = MoleculeDataset([test_data[i] for i in valid_indices])

    print(f'Test size = {len(test_data):,}')
    # Predict with each model individually and sum predictions
    if config.dataset_type == 'multiclass':
        sum_preds = np.zeros((len(test_data), num_tasks, config.multiclass_num_classes))
    else:
        sum_preds = np.zeros((len(test_data), num_tasks))
    print(f'Predicting with an ensemble of {len(config.checkpoint_paths)} models')
    for checkpoint_path in tqdm(config.checkpoint_paths, total=len(config.checkpoint_paths)):
        # Load model
        model = load_checkpoint(checkpoint_path)
        model_preds = predict(
            model=model,
            data=test_data,
            batch_size=config.batch_size,
        )
        sum_preds += np.array(model_preds)

    avg_preds = sum_preds / len(config.checkpoint_paths)
    avg_preds = avg_preds.tolist()

    # Put Nones for invalid smiles
    full_preds = [None] * len(full_data)
    for i, si in enumerate(valid_indices):
        full_preds[si] = avg_preds[i]
    avg_preds = full_preds
    
    avg_preds = np.array(avg_preds).reshape(-1)

    return avg_preds
