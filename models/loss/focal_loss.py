from turtle import forward
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='none') -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce_loss(input, target)

        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        elif self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        
        return focal_loss