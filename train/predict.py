# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from cmpnn.train import prediction
from cmpnn.data.utils import load_data

def predict(df):
    pred = prediction(df)
    return pred

if __name__ == '__main__':
    fp = '../../input/01_logP.csv'
    df = pd.read_csv(fp)
    tmp = load_data(df)
    df = pd.DataFrame({'smiles':tmp.smiles(), 'label':tmp.targets()})
    df['label'] = df['label'].apply(lambda x: x[0])
    del tmp
    print(df.head())
    assert df.shape[1] == 2
    pred = predict(df)
    df[df.columns[1]] = pred
    df.to_csv(f'./pred_{fp.split("/")[-1].split(".")[0]}.csv', index=False)