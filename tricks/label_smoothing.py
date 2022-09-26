import torch
import torch.nn as nn
import torch.nn.functional as F

class LSR(nn.Module):
    def __init__(self, n_classes=10, eps=0.1):
        super(LSR, self).__init__()
        self.n_classes = n_classes
        self.eps = eps

    def forward(self, outputs, labels):
        # labels.shape: [b,]
        assert outputs.size(0) == labels.size(0)
        n_classes = self.n_classes
        one_hot = F.one_hot(labels, n_classes).float()
        mask = ~(one_hot > 0)
        smooth_labels = torch.masked_fill(one_hot, mask, self.eps / (n_classes - 1))
        smooth_labels = torch.masked_fill(smooth_labels, ~mask, 1 - self.eps)
        ce_loss = torch.sum(-smooth_labels * F.log_softmax(outputs, 1), dim=1).mean()
        # ce_loss = F.nll_loss(F.log_softmax(outputs, 1), labels, reduction='mean')
        return ce_loss

def labelsmoothing():
    model=LSR(n_classes=2,eps=0.1)
    return model