import torch
import torch.nn as nn
import numpy as np

def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if len(true.shape) == 2:
        true = true.unsqueeze(0).unsqueeze(0)
    elif len(true.shape) == 3:
        true = true.unsqueeze(1)
        
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = nn.functional.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


class Loss(nn.Module):
    def __init__(self, num_classes=4, lmbda=10, epsilon=10e-6):
        super(ACLoss, self).__init__()
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.channels = num_classes 
        self.CELoss = nn.CrossEntropyLoss().cuda()
        
    def forward(self, pred, target, epoch=0):
        ce = self.CELoss(pred, target.long())
        dice = dice_loss(target.long(), pred)
        return dice + ce

    def index_to_bitmask(self, target):
        assert len(target.shape) == 3 #NHW
        
        target = target.unsqueeze(1) #NCHW
        tmp = []    
        for i in range(self.channels):
            tmp.append((target==i).long())
        
        return torch.cat(tmp, 1)
