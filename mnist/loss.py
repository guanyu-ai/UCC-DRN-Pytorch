import torch
import torch.nn as nn
import torch.nn.functional as F
import ot

class WassersteinLoss(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, input, targets):
        input = input.transpose(1,0)
        targets = targets.transpose(1,0)
        return ot.wasserstein_1d(input, targets).mean()