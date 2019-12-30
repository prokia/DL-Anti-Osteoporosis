# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:56:35 2019

@author: SY
"""
import torch

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    
    # padding
    target[index==0] = 0
    
    return target