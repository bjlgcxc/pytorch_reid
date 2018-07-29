import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from .metric import *

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, metric=None, margin=None, scalar=None, droprate=0):
        super(ClassBlock, self).__init__()

        self.metric = metric
        self.droprate = droprate
        self.bn = nn.BatchNorm1d(input_dim)
        self.relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(droprate)

        if metric == 'linear':
            self.classifier = nn.Linear(input_dim, class_num)
            print('linear classifier')
        else:
            args = {}
            if margin is not None:
                args['m'] = margin
            if scalar is not None:
                args['s'] = scalar
            if metric == 'cosface':
                self.classifier = AddMarginProduct(input_dim, class_num, **args)
            elif metric == 'arcface':
                self.classifier = ArcMarginProduct(input_dim, class_num, **args)
            elif metric == 'sphereface':
                self.classifier = SphereProduct(input_dim, class_num, **args)

    def forward(self, x, y):
        x = self.bn(x)
        x = self.relu(x)
        if self.droprate > 0:
            x = self.dropout(x)
        if y is None or self.metric=='linear':
            x = self.classifier(x)
        else:
            x = self.classifier(x, y)
        return x


class Backbone(nn.Module):
    def __init__(self, feat_size):
        super(Backbone, self).__init__()

        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model.layer4[0].downsample[0].stride = (1,1)      
        self.model.layer4[0].conv2.stride = (1,1)
        
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.fc = nn.Linear(1024, feat_size)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.'''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        c1 = self.model.conv1(x)
        c1 = self.model.bn1(c1)
        c1 = self.model.relu(c1)
        c1 = self.model.maxpool(c1)

        c2 = self.model.layer1(c1)
        c3 = self.model.layer2(c2)
        c4 = self.model.layer3(c3)
        c5 = self.model.layer4(c4)


        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        # Smooth
        p5 = self.avgpool(p5)
        p4 = self.avgpool(p4)
        p3 = self.avgpool(p3)
        p2 = self.avgpool(p2)

        x = torch.cat([p5, p4, p3, p2], 1)
        x = torch.squeeze(x)
        x = self.fc(x)

        return x


class FPN50(nn.Module):
    def __init__(self, class_num, feat_size, metric=None, margin=None, scalar=None, droprate=0):
        super(FPN50, self).__init__()
        self.model = Backbone(feat_size)
        self.classifier = ClassBlock(feat_size, class_num, metric, margin, scalar, droprate)

    def forward(self, x, y=None):
        feat = self.model(x)
        x = self.classifier(feat, y)

        return feat, x
