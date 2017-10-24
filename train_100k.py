from __future__ import print_function, absolute_import

import exchangable_tensor.layers
from exchangable_tensor.losses import mse
from data import df_to_matrix, get_mask
import pandas as pd
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


Encoder = exchangable_tensor.layers.Encoder
Decoder = exchangable_tensor.layers.Decoder

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
train = pd.read_csv("./data/ml-100k/u1.base", sep="\t", names=r_cols, encoding='latin-1')
validation = pd.read_csv("./data/ml-100k/u1.test", sep="\t", names=r_cols, encoding='latin-1')
train = df_to_matrix(train, 943, 1682)
validation = df_to_matrix(validation, 943, 1682)
#train[train>0] -= 3.5
#validation[validation>0] -= 3.5

def prep_var(x):
    return Variable(torch.from_numpy(np.array(x.toarray(), dtype="float32").reshape(x.shape[0], x.shape[1], 1)))

enc = Encoder(1, [10, 10], functions="mean", embedding_pool="mean")
dec = Decoder(10*2, [1], functions="mean")

pars = [i for i in enc.parameters()] + [i for i in dec.parameters()]
#for p in pars:
#    if len(p.size()) > 1:
#        nn.init.xavier_uniform(p, gain=nn.init.calculate_gain('relu'))
#    else:
#`       nn.init.constant(p, 0.01)

optimizer = torch.optim.Adam(pars)

train_x = prep_var(train)
train_mask = Variable(torch.from_numpy(get_mask(train)))
val_x = prep_var(validation)
val_mask = Variable(torch.from_numpy(get_mask(validation)))

epochs = 1000
for ep in xrange(epochs):
    optimizer.zero_grad()
    embeddings = enc(train_x, train_mask)
    y_hat = dec(embeddings, train_mask)
    loss = mse(y_hat, train_x, train_mask)
    reg_loss = 0
    for p in pars:
        reg_loss += torch.sum(torch.pow(p, 2))
    loss += 0.00001 * reg_loss
    loss.backward()
    optimizer.step()
    if ep % 1 == 0:
        val_loss = np.sqrt(mse(dec(enc(train_x, train_mask), val_mask), val_x, val_mask).data[0])
    print('Train Epoch: {}, Loss: {:.6f}, Val_loss: {:.6f}'.format(ep, np.sqrt(loss.data[0]), val_loss))
