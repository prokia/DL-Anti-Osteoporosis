# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 13:13:22 2019

@author: SY
"""

import pandas as pd
pd.set_option('display.max_columns', 12)

ans = pd.read_csv('./answer.csv')
ans.columns = ['smiles', 'label']
ans['label'] = 1

ans_bad = pd.read_csv('./answer_bad.csv')
ans_bad.columns = ['smiles', 'label']
ans_bad['label'] = 0

ans = ans.append(ans_bad).reset_index(drop=True)

pred = pd.read_csv('./test_pred.csv')
pred.columns = ['smiles', 'p_np']

pred = pred.sort_values(by='smiles', ascending=False).reset_index(drop=True)

pred = pred.sort_values(by='p_np', ascending=False).reset_index(drop=True)
pred['rank'] = pred.index.copy()+1

df = pred.merge(ans, on='smiles', how='left')
df = df.fillna(-1)

df = df.sort_values(by=['label', 'rank'], ascending=False)

df = df.drop_duplicates('smiles').reset_index(drop=True)
print(df[['smiles', 'rank', 'label']].head(50))
df.to_csv('./test_rank.csv', index=False)
