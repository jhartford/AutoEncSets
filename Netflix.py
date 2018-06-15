import time
import os
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from exchangable_tensor.sp_layers import SparseExchangeable, SparseSequential
from data import prep, collate_fn, CompletionDataset

from data.loader import IndexIterator
from data.samplers import ConditionalSampler, UniformSampler
import data.recsys

def prep_data(x, requires_grad=False):
    '''
    Helper function for setting up data variables
    '''
    x = Variable(x, requires_grad=requires_grad)
    if use_cuda:
        x = x.cuda()
    return x

def mask_inputs(batch, percent_train = 0.15):
    '''
    Mask inputs by setting some subset of the ratings to 0.
    '''
    input = batch['input']
    # indicator == 0 if training example, 1 if validation example and 2 if test example
    indicator = batch['indicator']
    
    if not isinstance(input, np.ndarray):
        input = input.numpy()
    if not isinstance(indicator, np.ndarray):
        indicator = indicator.numpy()
    
    # during training we set some percent of the training ratings to 0.
    if percent_train > 0.:
        # set validation and test ratings to 0.
        input[indicator == 2] = input[indicator == 2] * 0.
        input[indicator == 1] = input[indicator == 1] * 0.
        
        # sample training ratings to set to 0.
        n_train = input.shape[0]
        idx = np.arange(n_train)
        drop = np.random.permutation(idx[indicator == 0])[0:int(percent_train * n_train)]
        input[drop] = input[drop] * 0.
        
        # prepare for pytorch by moving numpy arrays to torch arrays
        batch["input"] = torch.from_numpy(input)
        for key in ["target", "index"]:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
        return batch, drop
    
    # during evaluation only the test ratings are set to 0.
    else:
        # set test ratings to zero.
        input[indicator == 2] = input[indicator == 2] * 0.
        idx = np.arange(input.shape[0])
        drop = idx[indicator == 2]
        
        # prepare for pytorch by moving numpy arrays to torch arrays
        batch["input"] = torch.from_numpy(input)
        for key in ["target", "index"]:
            if isinstance(batch[key], np.ndarray):
                batch[key] = torch.from_numpy(batch[key])
        return batch, drop
    
def one_hot(values):
    unique, inv = np.unique(values, return_inverse=True)
    n_unique = unique.shape[0]
    oh = np.zeros((values.shape[0], n_unique), dtype="float32")
    oh[np.arange(values.shape[0]), inv] = 1.
    return oh

use_cuda = True
enc = torch.load('100k_model.pt')
if use_cuda:
    enc.cuda()
data = np.load("data/netflix6m.npz")
netflix = {}
netflix['target'] = data['mat_values_all'][0:31500000, ...]
netflix["input"] = one_hot(netflix['target'])
netflix["index"] = data['mask_indices_all'][0:31500000, ...]
netflix["indicator"] = data['mask_tr_val_split'][0:31500000, ...]
full_batch, drop = mask_inputs(netflix, 0.)
target = prep_data((full_batch["target"]).long())
input = prep_data(full_batch["input"])
index = prep_data(full_batch["index"])
output = enc.cached_forward(input, index, batch_size=100000)
test_loss = expected_mse(output[drop,:].cuda(), target.squeeze(1).float()[drop].cuda())
print np.sqrt(test_loss.item())
