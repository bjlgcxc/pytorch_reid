import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1).cpu())
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index.cpu(), ones).cuda()


class AMSoftmax(nn.Module):
    def __init__(self, num_classes=10, feat_dim=512, m=0.35, s=30):
        super(AMSoftmax, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = m
        self.s = s
        self.kernel = nn.Parameter(torch.FloatTensor(self.feat_dim, self.num_classes).cuda())
        nn.init.kaiming_uniform(self.kernel.t())

    def forward(self, feat, labels):
        feat_norm = F.normalize(feat, p=2, dim=1)
        kernel_norm = F.normalize(self.kernel, p=2, dim=0)

        cos_theta = torch.mm(feat_norm, kernel_norm)
        cos_theta = torch.clamp(cos_theta, min=-1, max=1)
        phi = cos_theta - self.m
        label_onehot = one_hot(labels, self.num_classes)
        adjust_theta = self.s * ( label_onehot * phi + (1 - label_onehot) * cos_theta )

        return adjust_theta
