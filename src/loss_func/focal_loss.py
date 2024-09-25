import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig

class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(reduction=reduction)
        self.gamma = gamma
        self.weight = torch.tensor(weight)

    def forward(self, input, target):
        # weight_ten= torch.tensor(self.weight)
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss