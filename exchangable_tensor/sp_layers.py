import torch
import torch.nn as nn
from torch.autograd import Variable

class SparsePool(nn.Module):
    '''
    Sparse pooling with lazy memory management. Memory is set with the initial mask, but 
    can be reallocated as needed by changing the mask.
    '''
    def __init__(self, mask, out_features, axis, out_size=None, keep_dims=True, eps=1e-9):
        super(SparsePool, self).__init__()
        self.eps = eps
        self.axis = axis
        self._mask = mask
        self.out_features = out_features
        if out_size is None:
            out_size = mask[:, axis].max().data[0] + 1
        self.out_size = out_size
        self.output = Variable(torch.zeros(out_size, out_features), volatile=False)
        
        self.norm = Variable(torch.zeros(out_size), volatile=False, requires_grad=False)
        
        if mask.data.is_cuda:
            self.output = self.output.cuda()
            self.norm = self.norm.cuda()
        self.norm = self.norm.index_add_(0, mask[:, axis], torch.ones_like(mask[:, axis].float())) + self.eps
    
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        '''
        Setter for changing the mask. If the mask changes, we recalculate the normalization terms
        and if necessary, resize memory allocation.
        '''
        self._mask.data = mask.data
        out_size = mask[:, self.axis].max().data[0] + 1
        if out_size != self.out_size:
            del self.output, self.norm
            self.output = Variable(torch.zeros(out_size, self.out_features), volatile=False)
            self.norm = Variable(torch.zeros(out_size), volatile=False, requires_grad=False)
            if mask.data.is_cuda:
                self.output = self.output.cuda()
                self.norm = self.norm.cuda()
        
        self.norm = torch.zeros_like(self.norm).index_add_(0, mask[:, self.axis],
                                         torch.ones_like(mask[:, self.axis].float())) + self.eps
        
    def forward(self, input):
        self.output = torch.zeros_like(self.output)
        output = torch.zeros_like(self.output).index_add_(0, 
                                                          self.mask[:, self.axis], 
                                                          input)
        return torch.index_select(output / self.norm[:, None].float(), 0, self.mask[:, self.axis])
        

def mean_pool(input, mask, axis=0, out_size=None, keep_dims=True, eps=1e-9):
    '''
    Sparse mean pooling. This function performs the same role as the class
    above but is approximately 15% slower. Kept in the code base because it
    is much more readable.
    '''
    if out_size is None:
        out_size = mask[:, axis].max().data[0] + 1
    # Sum across values
    out = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    out = out.index_add_(0, mask[:, axis], input)
    
    # Normalization
    norm = Variable(input.data.new(out_size, input.shape[1]).fill_(0.))
    norm = norm.index_add_(0, mask[:, axis], torch.ones_like(input)) + eps
    if keep_dims:
        return torch.index_select(out / norm, 0, mask[:, axis])
    else:
        return out / norm

class SparseExchangeable(nn.Module):
    """
    Sparse exchangable matrix layer
    """

    def __init__(self, in_features, out_features, mask, bias=True, activ=None):
        super(SparseExchangeable, self).__init__()
        self._mask = mask
        self.linear = nn.Linear(in_features=in_features * 4,
                                out_features=out_features,
                                bias=bias)
        self.row_pool = SparsePool(self._mask, in_features, 0)
        self.col_pool = SparsePool(self._mask, in_features, 1)
        if activ is None:
            self.activ = lambda x: x
        else:
            self.activ = activ

    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        self.row_pool.mask = mask
        self.col_pool.mask = mask
        self._mask = mask
    
    def forward(self, input):
        row_mean = self.row_pool(input)
        col_mean = self.col_pool(input)
        both_mean = torch.mean(input, dim=0).expand_as(input)
        stacked = torch.cat([input, row_mean, col_mean, both_mean], dim=1)
        out = self.activ(self.linear(stacked))
        del stacked, row_mean, col_mean
        return out

class SparseFactorize(nn.Module):
    """
    Sparse factorization layer
    """

    def forward(self, input, mask):
        row_mean = mean_pool(input, mask, 0)
        col_mean = mean_pool(input, mask, 1)
        return torch.cat([row_mean, col_mean], dim=1)#, mask


class MaskedSequential(nn.Module):
    def __init__(self, mask, *args):
        super(MaskedSequential, self).__init__()
        self._mask = mask
        self.layers = nn.ModuleList(list(args))
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, mask):
        for l in self.layers:
            l.mask = mask
        self._mask = mask
    
    def forward(self, input):
        out = input
        for l in self.layers:
            out = l(out)
        return out
