import pandas as pd
import numpy as np

import time

import torch.nn as nn
import torch
from torch.autograd import Variable
from exchangable_tensor.sp_layers import SparseExchangeable, SparseFactorize, MaskedSequential
from data import prep, collate_fn, CompletionDataset
import data.recsys
from tqdm import tqdm

def mask_inputs(batch, perc = 0.15):
    input = batch['input']
    indicator = batch['indicator']
    
    if not isinstance(input, np.ndarray):
        input = input.numpy()
    if not isinstance(indicator, np.ndarray):
        indicator = indicator.numpy()
    
    if perc > 0.:
        input[indicator == 2] = input[indicator == 2] * 0.
        input[indicator == 1] = input[indicator == 1] * 0.
        idx = np.arange(input.shape[0])
        drop = np.random.permutation(idx[indicator == 0])[0:int(perc * input.shape[0])]
        input[drop] = input[drop] * 0.
        batch["input"] = torch.from_numpy(input)
        return batch, drop
    else:
        input[indicator == 2] = input[indicator == 2] * 0.
        batch["input"] = torch.from_numpy(input)
        return batch

data = data.recsys.ml100k(0.)
epochs = 100

mask = prep(data.mask, dtype="int").cuda()
enc = MaskedSequential(mask, 
                       SparseExchangeable(5,250, mask, activ=nn.LeakyReLU()), 
                       #SparseExchangeable(250,250, mask, activ=nn.LeakyReLU()),
                       #SparseExchangeable(250,250, mask, activ=nn.LeakyReLU()),
                       #SparseExchangeable(250,250, mask, activ=nn.LeakyReLU()),
                       SparseExchangeable(250,250, mask, activ=nn.LeakyReLU()),
                       SparseExchangeable(250,5, mask, activ=nn.LeakyReLU())
                   )

enc.cuda()

dataloader = torch.utils.data.DataLoader(data, num_workers=1, collate_fn=collate_fn, 
                                         batch_size=80000, shuffle=True 
                                        )
loss = torch.nn.CrossEntropyLoss()
#mse = torch.nn.MeanSquareError()
optimizer = torch.optim.Adam(enc.parameters(), lr=0.005)

t = time.time()
for epoch in xrange(epochs):
    for i_batch, sampled_batch in tqdm(enumerate(dataloader)):
        sampled_batch, drop = mask_inputs(sampled_batch)
        target = Variable((sampled_batch["target"] - 1).long()).cuda()
        input = Variable(sampled_batch["input"], requires_grad=False).cuda()
        mask = Variable(sampled_batch["mask"], requires_grad=False).cuda()
        enc.mask = mask
        optimizer.zero_grad()
        output = enc(input)
        l = loss(output[drop, :], target.squeeze(1)[drop])
        l.backward()
        optimizer.step()
    
    full_batch = mask_inputs(data[0:100000], 0.)
    #target = Variable((full_batch["target"] - 1).long()).cuda()
    #input = Variable(full_batch["input"], requires_grad=False).cuda()
    #mask = Variable(full_batch["mask"], requires_grad=False).cuda()
    #test_loss = mse(output[drop, :], target.squeeze(1)[drop])
    tqdm.write("%s" % l.cpu().data.numpy()[0])

sec_per_ep = (time.time() - t) / epochs
print "Time per epoch: %1.3f" % (sec_per_ep)
print "Est total time: %1.3f" % (sec_per_ep * 10000 / 60 / 60)
