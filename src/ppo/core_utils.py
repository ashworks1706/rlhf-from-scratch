import torch

def masked_mean(data, mask, dim=None, eps=1e-8):
    """
    Helper function for computing the mean of masked values.
    """
    data = data * mask
    if dim is not None:
        return data.sum(dim=dim) / (mask.sum(dim=dim) + eps)
    else:
        return data.sum() / (mask.sum() + eps)

def masked_var(data, mask, dim=None):
    """
    Helper function for computing the variance of masked values.
    """
    mean = masked_mean(data, mask, dim=dim)
    centered_vals = data - mean
    var = masked_mean(centered_vals ** 2, mask, dim=dim)

def masked_whiten(data, mask, dim=None, eps=1e-8, shift_mean=True):
    """
    Used in the reward whitening trick, which is an affine transformation on
    the token-level reward such that STD of the batch is 1. 
    The batch mean is preserved, though.
    """
    mean, var = masked_mean(data, mask), masked_var(data, mask)
    whitened = (data - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        whitened += mean
    return whitened
