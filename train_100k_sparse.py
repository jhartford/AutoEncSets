import time
import argparse
from tqdm import tqdm

import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable

from exchangable_tensor.sp_layers import SparseExchangeable, SparseSequential
from data import prep, collate_fn, CompletionDataset
import data.recsys
from data.loader import IndexIterator
from data.samplers import ConditionalSampler, UniformSampler

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, help='number of epochs to train for', default=5000)
parser.add_argument('--nocuda', action='store_true', help='disables cuda')
args = parser.parse_args()
use_cuda = not args.nocuda

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

data = data.recsys.ml100k(0.)
dataloader = torch.utils.data.DataLoader(data, num_workers=1, 
                                         collate_fn=collate_fn, 
                                         batch_size=80000, shuffle=True 
                                        )
index = prep(data.index, dtype="int")
if use_cuda:
    index = index.cuda()

# build model
enc = SparseSequential(index, 
                       SparseExchangeable(5,150, index), 
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(150,150, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(150,150, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(150,150, index),
                       nn.LeakyReLU(),
                       torch.nn.Dropout(p=0.5),
                       SparseExchangeable(150,5, index)
                   )
if use_cuda:
    enc.cuda()
optimizer = torch.optim.Adam(enc.parameters(), lr=0.005)

# Prepare cross entropy loss
ce = torch.nn.CrossEntropyLoss(reduce=False)
def masked_loss(output, target, drop, alpha=0.1):
    mask = torch.zeros_like(target)
    mask[drop] = 1
    mask = mask.float()
    ce_loss = ce(output, target)
    return ((1-alpha) * mask * ce_loss / mask.sum() + alpha * (1-mask) * ce_loss / (1-mask).sum()).sum()

# Prepare mean square error loss
mse = torch.nn.MSELoss()
values = prep_data(torch.arange(1,6)[None,:])
softmax = torch.nn.Softmax(dim=1)
def expected_mse(output, target):
    output = softmax(output)
    y = (output * values).sum(dim=1)
    return mse(y, target)

t = time.time()
sampler = UniformSampler(80000, data)

for epoch in xrange(args.epochs):
    # Training steps
    enc.train()
    iterator = IndexIterator(data, 80000, sampler, n_workers=1, 
                             return_last=True, epochs=1)
    for sampled_batch in tqdm(iterator):
        sampled_batch, drop = mask_inputs(sampled_batch)
        target = prep_data((sampled_batch["target"] - 1).long())
        input = prep_data(sampled_batch["input"])
        index = prep_data(sampled_batch["index"])
        enc.index = index
        optimizer.zero_grad()
        output = enc(input)
        l = masked_loss(output, target.squeeze(1), drop)
        l.backward()
        optimizer.step()
    
    # Evaluation
    enc.eval()
    full_batch, drop = mask_inputs(data[np.arange(100000)], 0.)
    target = prep_data((full_batch["target"]).long())
    input = prep_data(full_batch["input"])
    index = prep_data(full_batch["index"])
    enc.index = index
    test_loss = expected_mse(enc(input)[drop,:], target.squeeze(1).float()[drop])
    tqdm.write("%d, %s, %s" % (epoch, l.cpu().data.numpy()[0], 
                           np.sqrt(test_loss.cpu().data.numpy()[0])))

sec_per_ep = (time.time() - t) / args.epochs
print("Time per epoch: %1.3f" % (sec_per_ep))
print("Est total time: %1.3f" % (sec_per_ep * 10000 / 60 / 60))
