import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class RecLoss(nn.Module):
    def __init__(self, eta):
        super(RecLoss, self).__init__()
        self.eta = eta

    def forward(self, Y_pred, Y_true):
        loss = torch.linalg.norm(Y_true-Y_pred)
        return self.eta*loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, device=None):
        super(FocalLoss, self).__init__()
        self.alpha = torch.LongTensor([alpha, 1-alpha]).to(device)
        self.gamma = gamma

    def forward(self, Y_pred, Y_true):
        BCE_loss = F.binary_cross_entropy_with_logits(Y_pred, Y_true, reduction='none')
        print('BCE_loss SHAPE: ', BCE_loss.shape)
        # Y_true = Y_true.type(torch.long)
        at = self.alpha.gather(0, Y_true.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

if __name__ == '__main__':
    device = torch.device('cuda:0')
    Y_pred = torch.randint(low=0, high=5, size=(256,63), dtype=float)
    Y_true = torch.randint(low=0, high=5, size=(256,63), dtype=float)

    loss = FocalLoss(class_num=63)
    r = loss(Y_pred, Y_true)
    print('over')