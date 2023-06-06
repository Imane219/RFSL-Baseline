import numpy as np
import torch
from torch import nn
import pdb

class AdaptiveLabelSmoothingLoss(nn.Module):
    def __init__(self, classes=10, gamma=0.0, dim=-1, minVal=0, maxVal=1.):
        super(AdaptiveLabelSmoothingLoss, self).__init__()
        self.confidence = gamma
        self.smoothing = 1. - gamma
        self.cls = classes
        self.dim = dim
        self.minVal = minVal
        self.maxVal = maxVal

    def forward(self, pred, target, kappa=None, history_label=None):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            if kappa is not None:
                true_dist.scatter_(1, target.data.unsqueeze(1), (self.confidence * kappa).unsqueeze(1), reduce='add')
            else:
                true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            if history_label is not None:
                if kappa is not None:
                    true_dist = true_dist + (1 - self.confidence * kappa)[:, None] * history_label
                else:
                    true_dist = true_dist + self.smoothing * history_label

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=64, smoothing=0.0, dim=-1):#smoothing=0.2
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)#unsqueeze(1),在第1维增加一个维度,(10)->(10,1);将confidence值按照target的index输入到dict中
        #比如batchsize=2,label:[3,8];true_dist:[[0.0222,0.0222,0.0222,0.8,0.0222,...,0.0222],[0.0222,...,0.8,0.0222]],即将每个hard label换成0.8和0.0222这样的分布
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))