import torch
import torch.nn as nn

class EPGNNLoss(nn.Module):
    def __init__(self, mag_weight=0.1):
        super().__init__()
        self.clf_criterion = nn.CrossEntropyLoss()
        self.reg_criterion = nn.MSELoss()
        self.mag_weight = mag_weight

    def forward(self, logits, mag_pred, y_true, mag_true):
        loss_clf = self.clf_criterion(logits, y_true.view(-1))
        
        mask = y_true.view(-1) == 1
        if mask.sum() > 0:
            loss_mag = self.reg_criterion(mag_pred[mask].squeeze(-1), mag_true[mask].squeeze(-1))
            return loss_clf + self.mag_weight * loss_mag
        else:
            return loss_clf
