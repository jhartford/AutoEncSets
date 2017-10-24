import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

def masked_max(x, dim, mask=None):
    '''
    Slightly hacky masked max...
    '''
    if mask is None:
        return torch.max(x, dim=dim)[0]
    m = x.min()
    maxes = torch.max(x - torch.mul(m, mask), dim=dim)[0] + m
    return maxes

def masked_mean(x, dim, mask=None, eps=1e-9):
    '''
    Masked mean
    '''
    if mask is None:
        return torch.mean(x, dim)
    else:
        norm = torch.sum(mask, dim=dim) + eps
        x_sum = torch.sum(x, dim=dim)
        out = x_sum / norm
        return out

class SetPool(nn.Module):
    '''
    Basic pooling operations. 
    '''
    def __init__(self, axis, function = "mean", expand=True):
        super(SetPool, self).__init__()
        self.expand = expand
        self._function_name = function
        self._axis_name = axis
        
        if isinstance(axis, str):
            self.dim = {"row":0, "column":1, "both":None}[axis]
        else:
            self.dim = axis
        
        if function == "max":
            self.function = masked_max
        elif function == "sum":
            self.function = lambda x, dim, mask: torch.sum(x, dim=dim)    
        elif function == "mean":
            self.function = masked_mean
        else:
            raise ValueError("Unrecognised function: %s" % function)

    def forward(self, input, mask=None):
        if self.dim is None:
            m, n, d = input.size()
            reshaped = input.view(m*n, d)
            if mask is not None:
                m = mask.view(m*n, 1)
            else:
                m = mask
            # get the output shapes right
            output = self.function(reshaped, dim=0, mask=m)
            if self.expand:
                output = output.unsqueeze(1).unsqueeze(1)
                output = output.permute(1, 2, 0)
        else:
            output = self.function(input, dim=self.dim, mask=mask)
            if self.expand:
                output = output.unsqueeze(1).permute(1-self.dim, self.dim, 2)
        if self.expand:
            return output.expand_as(input)
        else:
            return output

class MatrixLinear(nn.Linear):
    '''
    Set-based linear feed-forward layer. Think of it as the
    set analog to a feed-forward layer in an MLP.
    '''
    def forward(self, input, mask=None):
        m, n, d = input.size()
        state = input.view((m*n, d))
        output = super(MatrixLinear, self).forward(state)
        output = output.view((m,n,self.out_features))
        if mask is not None:
            output = torch.mul(output, mask)
        return output

class MatrixLayer(nn.Module):
    '''
    Set layers are linear layers with pooling. Pooling operations
    are defined above.
    '''
    def __init__(self, in_features, out_features, pooling = "mean", 
                 axes=["row", "column", "both"]):
        super(MatrixLayer, self).__init__()

        # build list of pooling functions
        pool = []
        for axis in axes:
            pool.append(SetPool(axis, pooling, expand=True))
        self.pool = pool
        self.linear = MatrixLinear(in_features * (1+len(pool)), out_features)

    def forward(self, input, mask=None):
        pooled = [p(input, mask) for p in self.pool]
        state = torch.cat([input] + pooled, dim=2)
        return self.linear(state, mask)
    
class Encoder(nn.Module):
    def __init__(self, input_dim, units, functions="mean", activation="relu", embedding_pool="max"):
        super(Encoder, self).__init__()
        if isinstance(functions, str):
            functions = [functions] * len(units)
        units = [input_dim] + units
        layers = []
        self.activation = activation
        for i in xrange(1, len(units)):
            axes = ["row", "column", "both"] if i < (len(units)-1) else []
            layers.append(MatrixLayer(units[i-1], units[i], functions[i-1], axes=axes))
        self.embeddings = [SetPool("row", embedding_pool, expand=True), 
                           SetPool("column", embedding_pool, expand=True)]
        self.layers = nn.ModuleList(layers)

    def forward(self, input, mask=None):
        state = input
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            state = layer(state, mask)
            if i == last:
                break
            if self.activation == "relu":
                state = F.relu(state)
                if mask is not None:
                    state = torch.mul(state, mask)
        return [f(state, mask) for f in self.embeddings]
    
class Decoder(nn.Module):
    def __init__(self, embedding_dim, units, functions="mean", activation="relu"):
        super(Decoder, self).__init__()
        if isinstance(functions, str):
            functions = [functions] * len(units)
        units = [embedding_dim] + units
        layers = []
        self.activation = activation
        for i in xrange(1, len(units)):
            layers.append(MatrixLayer(units[i-1], units[i], functions[i-1]))
        self.layers = nn.ModuleList(layers)

    def forward(self, input_list, mask=None):
        state = torch.cat(input_list, dim=2)
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            state = layer(state, mask)
            if i == last:
                break
            if self.activation == "relu":
                state = F.relu(state)
                if mask is not None:
                    state = torch.mul(state, mask)
        return state
