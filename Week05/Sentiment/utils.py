import torch
from torch.autograd import Variable


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def detach(x):
    """Wraps hidden states in new variables, to detach them from their history."""
    if type(x) == Variable:
        return Variable(x.data)
    else:
        return tuple(detach(v) for v in x)
