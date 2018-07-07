import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import models
from torch.autograd import Variable


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
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
    def __init__(self, input_dim, class_num, dropout=True, relu=True, num_bottleneck=512):
        super(ClassBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier

    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x


class FPN101(nn.Module):
    def __init__(self, class_num):
        super(FPN101, self).__init__()
        self.class_num = class_num

        self.model = models.resnet50(pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

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

        # print(p5.size())

        return torch.cat([p5, p4, p3, p2], 1)


class FPN(nn.Module):
    def __init__(self, class_num, test=False):
        super(FPN, self).__init__()
        self.class_num = class_num
        self.test = test

        self.model = FPN101(class_num)
        self.classifier = ClassBlock(1024, self.class_num)

    def forward(self, x):
        x = self.model(x)
        x = torch.squeeze(x)
        if self.test:
            return x
        else:
            return x, self.classifier(x)


if __name__=='__main__':
    fpn = FPN(751)
    #print(fpn)
    #output = fpn(Variable(torch.randn(8, 3, 256, 128)))
    #print(output)
    #print(output[0].size())
    #print(output)