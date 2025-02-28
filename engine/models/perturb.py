import torch
import torch.nn as nn
import torch.nn.functional as F


def feat_perturbation(xt, xts, xst):
    gating_param = nn.Parameter(torch.ones(xt.shape[1]), requires_grad= True)
    gating_param = gating_param.to('cuda:0')
    gating = (gating_param).view(1,-1,1,1)
    attn = (1.-torch.sigmoid(gating)) * xts + torch.sigmoid(gating) * xst

    xt = xt + (attn * 0.01)
    return xt