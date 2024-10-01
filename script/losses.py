import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def cross_entropy_loss(input, target, eps=1e-8):
    # input = torch.clamp(input, eps, 1 - eps)
    input = F.softmax(input ,dim=1)
    loss = torch.mean(-target * torch.log(input+eps))
    return loss
