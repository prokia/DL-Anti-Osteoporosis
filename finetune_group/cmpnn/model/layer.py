# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 19:13:29 2019

@author: SY
"""
from cmpnn import config
import warnings
warnings.filterwarnings('ignore')
import torch
from torch import nn
import torch.nn.functional as F
import math
from cmpnn.data.feature import mol2graph
from cmpnn.nn_utils import index_select_ND

def build_model() -> nn.Module:
    output_size = config.num_tasks
    config.output_size = output_size
    if config.dataset_type == 'multiclass':
        config.output_size *= config.multiclass_num_classes

    model = Model(classification=config.dataset_type == 'classification', 
                  multiclass=config.dataset_type == 'multiclass')

    initialize_weights(model)

    return model


class Model(nn.Module):
    def __init__(self, classification: bool, multiclass: bool):
        super(Model, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.encoder = MP()
        self.create_ffn()


    def create_ffn(self):
        self.multiclass = config.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = config.multiclass_num_classes
        first_linear_dim = config.hidden_size * 1

        dropout = nn.Dropout(config.dropout)
        act = nn.ReLU()

        if config.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, config.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, config.ffn_hidden_size)
            ]
            for _ in range(config.ffn_num_layers - 2):
                ffn.extend([
                    act,
                    dropout,
                    nn.Linear(config.ffn_hidden_size, config.ffn_hidden_size),
                ])
            ffn.extend([
                act,
                dropout,
                nn.Linear(config.ffn_hidden_size, config.output_size),
            ])

        self.ffn = nn.Sequential(*ffn)

    def forward(self, *input):
        output = self.ffn(self.encoder(*input))

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes))
            if not self.training:
                output = self.multiclass_softmax(output)

        return output


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.atom_dim = config.atom_dim
        self.bond_dim = config.bond_dim
        self.hidden_size = config.hidden_size
        self.bias = config.bias
        self.depth = config.depth
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.act_func = nn.ReLU()

        # Input
        input_dim = self.atom_dim
        self.W_i_atom = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        input_dim = self.bond_dim
        self.W_i_bond = nn.Linear(input_dim, self.hidden_size, bias=self.bias)
        
        w_h_input_size_bond = self.hidden_size
        self.W_h_bond = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
        
        self.W_o = nn.Linear(
                self.atom_dim + 
                self.hidden_size,
                self.hidden_size)
        
        self.gru = BatchGRU(self.atom_dim + self.hidden_size)
        
        

    def forward(self, mol_graph):
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
        try:
            f_atoms, f_bonds, a2b, b2a, b2revb, bonds = (
                    f_atoms.cuda(), f_bonds.cuda(), 
                    a2b.cuda(), b2a.cuda(), b2revb.cuda(), 
                    bonds.cuda())
        except:
            pass
        
        # Input
        input_atom = self.W_i_atom(f_atoms)  # num_atoms x hidden_size
        input_atom = self.act_func(input_atom)
        
        input_bond = self.W_i_bond(f_bonds)  # num_bonds x hidden_size
        message_bond = self.act_func(input_bond)
        input_bond = self.act_func(input_bond)
        
        # mp
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            rev_message = message_bond[b2revb]
            message_bond = agg_message[b2a] - rev_message

            message_bond = self.W_h_bond(message_bond)
            message_bond = self.act_func(input_bond + message_bond)  
            message_bond = self.dropout_layer(message_bond)
            
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.gru(torch.cat([agg_message*input_atom, f_atoms],1), a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))
        atom_hiddens = self.dropout_layer(atom_hiddens)
        
        # Readout
        mol_vec = []
        for i, (a_start, a_size) in enumerate(a_scope):
            hidden = atom_hiddens.narrow(0, a_start, a_size)
            mol_vec.append(hidden.max(dim=0)[0])

        return torch.stack(mol_vec, dim=0)

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru  = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, 
                           bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size), 
                                1.0 / math.sqrt(self.hidden_size))


    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        # padding
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            cur_message = torch.nn.ZeroPad2d((0,0,0,MAX_atom_len-cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))
            
        message_lst = torch.cat(message_lst, 0)
        hidden_lst  = torch.cat(hidden_lst, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)
        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)
        message = torch.cat([message.narrow(0, 0, 1), cur_message_unpadding], 0)
        return message


class MP(nn.Module):
    def __init__(self):
        super(MP, self).__init__()
        self.atom_dim = config.atom_dim
        self.bond_dim = config.bond_dim
        self.encoder = Encoder()

    def forward(self, inputs) -> torch.FloatTensor:
        inputs = mol2graph(inputs)
        output = self.encoder.forward(inputs)

        return output


def initialize_weights(model: nn.Module):
    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)