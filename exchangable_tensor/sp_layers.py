import torch
import torch.nn as nn
from torch.autograd import Variable
from itertools import combinations

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
    _, valid_index = np.unique(index, axis=0, return_inverse=True)
    return valid_index

def prepare_global_index(index, axes=None):
    if axes is None:
        axes = subsets(index.shape[1])
    return np.concatenate([to_valid_index(index[:, ax])[:, None] for ax in axes], axis=1)

class SparsePool(nn.Module):
    '''
    Sparse pooling with lazy memory management. Memory is set with the initial index, but 
    can be reallocated as needed by changing the index.

    Caching deals with the memory limitations of these models by computing the pooling layers on
    CPU memory. A typical forward pass still uses batches on the GPU but pools on the CPU 
    (see SparseExchangable below).
    '''
    def __init__(self, index, out_features, out_size=None, keep_dims=True, eps=1e-9, cache_size=None):
        super(SparsePool, self).__init__()
        self.eps = eps
        self._index = index
        self.out_features = out_features
        self.keep_dims = keep_dims
        if out_size is None:
            out_size = int(index.max() + 1)
        self.out_size = out_size
        self.output = Variable(torch.zeros(out_size, out_features), volatile=False)
        
        self.norm = Variable(torch.zeros(out_size), volatile=False, requires_grad=False)
        
        #if index.data.is_cuda:
        self.output = self.output.to(index.device)
        self.norm = self.norm.to(index.device)
        self.norm = self.norm.index_add_(0, index, torch.ones_like(index.float())) + self.eps
        self.cache_size = cache_size
    
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
        out_size = int(index.max() + 1)
        if out_size != self.out_size:
            del self.output, self.norm
            self.output = Variable(torch.zeros(out_size, self.out_features), volatile=False)
            self.norm = Variable(torch.zeros(out_size), volatile=False, requires_grad=False)
            if index.data.is_cuda:
                self.output = self.output.cuda()
                self.norm = self.norm.cuda()
            self.out_size = out_size
        
        self.norm = torch.zeros_like(self.norm).index_add_(0, index,
                                         torch.ones_like(index.float())) + self.eps
        
    def zero_cache(self):
        '''
        We incrementally compute the pooled representation in batches, so we need a way of clearing
        the cached representation.
        '''
        if self.cache_size is None:
            raise ValueError("Must specify a cache size if using a cache")
        self._cache = torch.zeros(self.cache_size, self.out_features)
        self._cache_norm = torch.zeros(self.cache_size) + self.eps
        
    def update_cache(self, input, index):
        '''
        Add a batch to the pooled representation
        '''
        self._cache = self._cache.index_add_(0, index.cpu(), input.cpu())
        self._cache_norm = self._cache_norm.index_add_(0, index.cpu(),
                                            torch.ones_like(index.cpu().float())) + self.eps
    
    def get_cache(self, index, keep_dims=True):
        '''
        Return the pooled representation.
        '''
        output = self._cache / self._cache_norm[:, None].float()
        if keep_dims:
            return torch.index_select(output, 0, index.cpu())
        else:
            return output
    
    def forward(self, input, keep_dims=None, cached=False, index=None):
        '''
        Regular forward pass.
        '''
        if index is None:
            index = self.index
        if keep_dims is None:
            keep_dims = self.keep_dims
        if cached:
            return self.get_cache(index, keep_dims)
        self.output = torch.zeros_like(self.output)
        output = torch.zeros_like(self.output).index_add_(0, 
                                                          index, 
                                                          input)
        if keep_dims:
            return torch.index_select(output / self.norm[:, None].float(), 
                                      0, index)
        else:
            return output / self.norm[:, None].float()

class SparseExchangeable(nn.Module):
    """
    Sparse exchangable matrix layer
    """
    def __init__(self, in_features, out_features, index, bias=True, cache_size=None):
        super(SparseExchangeable, self).__init__()
        self._index = index
        self.pooling = nn.ModuleList([SparsePool(index[:, i], in_features) for i in range(index.shape[1])])
        self.linear = nn.Linear(in_features=in_features * (index.shape[1] + 2),
                                out_features=out_features,
                                bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.cache_size = cache_size

    def zero_cache(self):
        self._cache = torch.zeros(self.cache_size, self.out_features)
    
    def update_cache(self, input, index, batch_size=10000):
        nnz = index.shape[0] # number of non-zeros
        cache_sizes = index.cpu().numpy().max(axis=0) + 1
        batch_size = min(batch_size, nnz)
        splits = max(nnz // batch_size, 1)
        for i, p in enumerate(self.pooling):
            p.cache_size = cache_sizes[i]
            p.zero_cache()
        for i in np.split(np.arange(nnz), splits):
            for j, p in enumerate(self.pooling):
                p.update_cache(input.cpu()[i, ...], index.cpu()[i, j])
        pooled = [p.get_cache(index.cpu()[:,i], keep_dims=True) for i, p in enumerate(self.pooling)]
        pooled += [torch.mean(input.cpu(), dim=0).expand_as(input)]
        stacked = torch.cat([input.cpu()] + pooled, dim=1)
        for i in np.split(np.arange(nnz), splits):
            self._cache[i,...] = self.linear(stacked[i,...].to(self.linear.weight.device)).cpu()
    
    def get_cache(self, idx=None):
        if idx is None:
            return self._cache
        else:
            return self._cache[idx, ...]
    
    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        for module, axis in zip(self.pooling, self.axes):
            sub_index = to_valid_index(index[:, axis])
            module.index = sub_index
        self._index = index
    
    def forward(self, input):
        pooled = [pool_axis(input) for pool_axis in self.pooling]
        pooled += [torch.mean(input, dim=0).expand_as(input)]
        stacked = torch.cat([input] + pooled, dim=1)
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
    
    def cached_forward(self, input, index, batch_size=10000):
        with torch.no_grad():
            state = input
            for i, layer in enumerate(self.layers):
                print("layer %d" % i)
                state.detach()
                if isinstance(layer, SparseExchangeable):
                    layer.cache_size = index.shape[0]
                    layer.zero_cache()
                    layer.update_cache(state, index, batch_size=batch_size)
                    state = layer.get_cache()
                else:
                    state = layer(state)
                del layer
            return state

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
