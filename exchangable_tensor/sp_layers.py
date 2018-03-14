import torch
import torch.nn as nn
from torch.autograd import Variable

class SparsePool(nn.Module):
    '''
    Sparse pooling with lazy memory management. Memory is set with the initial index, but 
    can be reallocated as needed by changing the index.
    '''
    def __init__(self, index, out_features, axis, out_size=None, keep_dims=True, eps=1e-9):
        super(SparsePool, self).__init__()
        self.eps = eps
        self.axis = axis
        self._index = index
        self.out_features = out_features
        if out_size is None:
            out_size = index[:, axis].max().data[0] + 1
        self.out_size = out_size
        self.output = Variable(torch.zeros(out_size, out_features), volatile=False)
        
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
        out_size = index[:, self.axis].max().data[0] + 1
        if out_size != self.out_size:
            del self.output, self.norm
            self.output = Variable(torch.zeros(out_size, self.out_features), volatile=False)
            self.norm = Variable(torch.zeros(out_size), volatile=False, requires_grad=False)
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
        return torch.index_select(output / self.norm[:, None].float(), 0, self.index[:, self.axis])
        

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

    def __init__(self, in_features, out_features, index, bias=True):
        super(SparseExchangeable, self).__init__()
        self._index = index
        self.linear = nn.Linear(in_features=in_features * 4,
                                out_features=out_features,
                                bias=bias)
        self.row_pool = SparsePool(self._index, in_features, 0)
        self.col_pool = SparsePool(self._index, in_features, 1)

    @property
    def index(self):
        return self._index
    
    @index.setter
    def index(self, index):
        self.row_pool.index = index
        self.col_pool.index = index
        self._index = index
    
    def forward(self, input):
        row_mean = self.row_pool(input)
        col_mean = self.col_pool(input)
        both_mean = torch.mean(input, dim=0).expand_as(input)
        stacked = torch.cat([input, row_mean, col_mean, both_mean], dim=1)
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
