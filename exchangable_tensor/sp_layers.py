import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import combinations
import numpy as np

def subsets(n, return_empty=False):
    '''
    Get all proper subsets of [0, 1, ..., n]
    '''
    sub = [i for j in range(n) for i in combinations(range(n), j)]
    if return_empty:
        return sub
    else:
        return sub[1:]

def to_valid_index(index):
    if isinstance(index, torch.tensor):
        index = index.numpy()
    _, valid_index = np.unique(index, axis=0, return_inverse=True)
    return torch.from_numpy(valid_index)

def append_features(index, interaction=None, row_values=None, col_values=None, dtype="float32"):
    '''
    Append features to the values matrix using the index to map to the correct dimension.

    Used when we have row or column features. Assumes that the index a zero-index (i.e. counts from zero).
    '''
    if interaction is None and row_values is None and col_values is None:
        raise Exception("Must supply at least one value array.")
    values = np.zeros((index.shape[0], 0), dtype=dtype)
    if interaction is not None:
        if len(interaction.shape) == 1:
            interaction = interaction[:, None]
        values = np.concatenate([values, interaction], axis=1)
    if row_values is not None:
        if len(row_values.shape) == 1:
            row_values = row_values[:, None]
        values = np.concatenate([values, row_values[index[:, 0], ...]], axis=1)
    if col_values is not None:
        if len(col_values.shape) == 1:
            col_values = col_values[:, None]
        values = np.concatenate([values, col_values[index[:, 1], ...]], axis=1)
    return values

class SparsePool(nn.Module):
    '''
    Sparse pooling with lazy memory management. Memory is set with the initial index, but 
    can be reallocated as needed by changing the index.
    '''
    def __init__(self, index, out_features, axis, out_size=None, keep_dims=True, 
                 normalize=True, eps=1e-9):
        super(SparsePool, self).__init__()
        self.eps = eps
        self.axis = axis
        self._index = index
        self.out_features = out_features
        self.keep_dims = keep_dims
        self.normalize = normalize
        if out_size is None:
            out_size = int(index[:, axis].max() + 1)
        self.out_size = out_size
        self.output = Variable(torch.zeros((out_size, out_features)), volatile=False)
        self.norm = Variable(torch.zeros(out_size), volatile=False, requires_grad=False)
        
        if index.data.is_cuda:
            self.output = self.output.cuda()
            self.norm = self.norm.cuda()
        self.norm = self.norm.index_add_(0, index[:, axis], torch.ones_like(index[:, axis].float())) + self.eps
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        '''
        Setter for changing the index. If the index changes, we recalculate the normalization terms
        and if necessary, resize memory allocation.
        '''
        self._index = index
        out_size = int(index[:, self.axis].max() + 1)
        if out_size != self.out_size:
            del self.output, self.norm
            self.output = Variable(torch.zeros((out_size, self.out_features)), volatile=False)
            self.norm = Variable(torch.zeros((out_size)), volatile=False, requires_grad=False)
            if index.data.is_cuda:
                self.output = self.output.cuda()
                self.norm = self.norm.cuda()
            self.out_size = out_size
        
        self.norm = torch.zeros_like(self.norm).index_add_(0, index[:, self.axis],
                                         torch.ones_like(index[:, self.axis].float())) + self.eps
        
    def forward(self, input):
        self.output = torch.zeros_like(self.output)
        output = torch.zeros_like(self.output).index_add_(0, 
                                                          self.index[:, self.axis], 
                                                          input)
        if self.normalize:
            output = output / self.norm[:, None].float()
        else:
            output = output
        
        if self.keep_dims:
            return torch.index_select(output, 
                                      0, self.index[:, self.axis])
        else:
            return output
        

def mean_pool(input, index, axis=0, out_size=None, keep_dims=True, eps=1e-9):
    '''
    Sparse mean pooling. This function performs the same role as the class
    above but is approximately 15% slower. Kept in the codebase because it
    is much more readable.
    '''
    if out_size is None:
        out_size = index[:, axis].max().data[0] + 1
    # Sum across values
    out = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    out = out.index_add_(0, index[:, axis], input)
    
    # Normalization
    norm = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    norm = norm.index_add_(0, index[:, axis], torch.ones_like(input)) + eps
    if keep_dims:
        return torch.index_select(out / norm, 0, index[:, axis])
    else:
        return out / norm

class SparseExchangeable(nn.Module):
    """
    Sparse exchangable matrix layer
    """

    def __init__(self, in_features, out_features, index, bias=True,
                 row_pool=True, col_pool=True, both_pool=True):
        super(SparseExchangeable, self).__init__()
        self._index = index
        n_pool = 1 + int(row_pool) + int(col_pool) + int(both_pool)
        self.linear = nn.Linear(in_features=in_features * n_pool,
                                out_features=out_features,
                                bias=bias)
        if row_pool:
            self.row_pool = SparsePool(self._index, in_features, 0)
        else:
            self.row_pool = None
        if col_pool:
            self.col_pool = SparsePool(self._index, in_features, 1)
        else:
            self.col_pool = None
        self.both_pool = both_pool

    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        if self.row_pool is not None:
            self.row_pool.index = index
        if self.col_pool is not None:
            self.col_pool.index = index
        self._index = index
    
    def forward(self, input):
        inputs = [input]
        if self.row_pool is not None:
            row_mean = self.row_pool(input)
            inputs.append(row_mean)
        if self.col_pool is not None:
            col_mean = self.col_pool(input)
            inputs.append(col_mean)
        if self.both_pool:
            both_mean = torch.mean(input, dim=0).expand_as(input)
            inputs.append(both_mean)
        stacked = torch.cat(inputs, dim=1)
        return self.linear(stacked)

class SparseFactorize(nn.Module):
    """
    Sparse factorization layer
    """

    def forward(self, input, index):
        row_mean = mean_pool(input, index, 0)
        col_mean = mean_pool(input, index, 1)
        return torch.cat([row_mean, col_mean], dim=1)#, index


class SparseSequential(nn.Module):
    def __init__(self, index, *args):
        super(SparseSequential, self).__init__()
        self._index = index
        self.layers = nn.ModuleList(list(args))
        
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        for l in self.layers:
            if hasattr(l, "index"):
                l.index = index
        self._index = index
    
    def forward(self, input):
        out = input
        for l in self.layers:
            out = l(out)
        return out

# Not used...

def mean_pool(input, index, axis=0, out_size=None, keep_dims=True, eps=1e-9):
    '''
    Sparse mean pooling. This function performs the same role as the class
    above but is approximately 15% slower. Kept in the codebase because it
    is much more readable.
    '''
    if out_size is None:
        out_size = index[:, axis].max().data[0] + 1
    # Sum across values
    out = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    out = out.index_add_(0, index[:, axis], input)
    
    # Normalization
    norm = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    norm = norm.index_add_(0, index[:, axis], torch.ones_like(input)) + eps
    if keep_dims:
        return torch.index_select(out / norm, 0, index[:, axis])
    else:
        return out / norm
