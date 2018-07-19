import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from .metric import * 

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, metric=None, margin=None, scalar=None):
        super(ClassBlock, self).__init__()
		
        self.metric = metric

        if metric == 'linear':
            self.classifier = nn.Linear(input_dim, class_num)
            weights_init_classifier(self.classifier)
        else:
            args = {}
            if margin:
                args['m'] = margin
            if scalar:
                args['s'] = scalar
            if metric == 'cosface':
                self.classifier = AddMarginProduct(input_dim, class_num, **args)
            elif metric == 'arcface':
                self.classifier = ArcMarginProduct(input_dim, class_num, **args)
            elif metric == 'sphereface':
                self.classifier = SphereProduct(input_dim, class_num, **args)

    def forward(self, x, y):
        if y is None or self.metric=='linear':
            x = self.classifier(x)
        else:
            x = self.classifier(x, y)

        return x

class Backbone(nn.Module):
    def __init__(self, feat_size, pretrained=True):
        super(Backbone, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        if not pretrained:
            weights_init_kaiming(model_ft)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1000, feat_size)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.squeeze(x)
        x = self.model.fc(x)
        x = self.linear(x)
        
        return x

class ResNet50(nn.Module):
    def __init__(self, class_num, feat_size, metric=None, margin=None, scalar=None, dropout=0):
        super(ResNet50, self).__init__()
        self.model = Backbone(feat_size)
        self.drop = dropout
        self.bn = nn.BatchNorm1d(feat_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = ClassBlock(feat_size, class_num, metric, margin, scalar)

    def forward(self, x, y=None):
        x = self.model(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.drop > 0:
            x = self.dropout(x)
		
        x = self.classifier(x, y)
        return x
