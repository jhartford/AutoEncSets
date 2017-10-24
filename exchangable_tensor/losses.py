import torch

def mse(predicted, target, mask=None):
    if mask is not None:
        return torch.sum(torch.pow(predicted - target, 2) * mask) / torch.sum(mask)
    else:
        return torch.mean(torch.pow(predicted - target, 2))

