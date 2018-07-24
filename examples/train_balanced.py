# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
import time
import os
import sys
import shutil
if not os.getcwd() in sys.path:
     sys.path.append(os.getcwd())
from models import ResNet50
from models.utils import StepLRScheduler
from utils.random_erasing import RandomErasing
import json
from data import BalancedSampler 

#########################################################################################################
# Options 
# --------
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--name', default='ResNet50', type=str, help='output model name')
parser.add_argument('--data_dir', default='../dataset/Market/pytorch', type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data')
parser.add_argument('--color_jitter', action='store_true', help='use color jitter in training')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--metric', default=None, type=str, help='metric, in [arcface, cosface, sphereface]')
parser.add_argument('--margin', default=None, type=float, help='margin')
parser.add_argument('--scalar', default=None, type=float, help='scalar')
parser.add_argument('--optim_type', default='SGD_Step', type=str, help='SGD_Step, SGD_warmup, Adam')
parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
parser.add_argument('--feat_size', default=1024, type=int, help='feature size')
##########################################################################################################


##########################################################################################################
# Init
# -------

opt = parser.parse_args()

data_dir = opt.data_dir
name = opt.name
metric = opt.metric
margin = opt.margin
scalar = opt.scalar
optim_type = opt.optim_type
dropout = opt.dropout
feat_size = opt.feat_size

transform_train_list = [
    transforms.Resize((288, 144), interpolation=3),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

transform_val_list = [
    transforms.Resize(size=(288, 144), interpolation=3),  # Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

if opt.erasing_p > 0:
    transform_train_list = transform_train_list + [RandomErasing(probability=opt.erasing_p, mean=[0.0, 0.0, 0.0])]

if opt.color_jitter:
    transform_train_list = [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1,
                                                   hue=0)] + transform_train_list
data_transforms = {
    'train': transforms.Compose(transform_train_list),
    'val': transforms.Compose(transform_val_list),
}

train_all = ''
if opt.train_all:
    train_all = '_all'
    print('train all.')

image_datasets = {}
image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train' + train_all),
                                               data_transforms['train'])
image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                             data_transforms['val'])

sampler = BalancedSampler(data_source=image_datasets['train'].imgs, num_instances=4)
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batchsize, 
             shuffle=False, sampler=sampler, num_workers=16),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=opt.batchsize,
             shuffle=True, num_workers=16)
}
dataset_sizes = {'train': len(sampler), 'val': len(image_datasets['val'])}
class_names = image_datasets['train'].classes

use_gpu = torch.cuda.is_available()

inputs, classes = next(iter(dataloaders['train']))


###########################################################################################################

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()
    
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler != None:
                    scheduler.step()
                    print('lr: {}'.format(scheduler.get_lr()[0]))
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0
            for data in dataloaders[phase]:
                inputs, labels = data
                inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                   
                optimizer.zero_grad()

                outputs = model(inputs, labels)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                if epoch % 10 == 9:
                    save_network(model, epoch)
                save_network(model, 'last')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model


def save_network(network, epoch_label):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join('./logs', name, save_filename)
    torch.save(network.cpu().state_dict(), save_path)
    if torch.cuda.is_available:
        network.cuda()


if __name__ == '__main__':

    model = ResNet50(len(class_names), feat_size, metric, margin, scalar, dropout).cuda()

    criterion = nn.CrossEntropyLoss()
    
	# SGD_Step
    if optim_type == 'SGD_Step':
        optimizer = optim.SGD(params=model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9, nesterov=True)
        lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        #lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 60, 70, 80], gamma=0.1)
    # SGD_Warmup
    elif optim_type == 'SGD_Warmup':
        lr_steps = [80, 100]
        init_lr = 5e-5
        gamma = 0.1
        warmup_lr = 1e-3
        warmup_steps = 20
        gap = warmup_lr - init_lr
        warmup_mults = [(init_lr + (i+1)*gap/warmup_steps) / (init_lr + i*gap/warmup_steps) for i in range(warmup_steps)]
        warmup_steps = list(range(warmup_steps))
        lr_mults = warmup_mults + [gamma]*len(lr_steps)
        lr_steps = warmup_steps + lr_steps
        optimizer = optim.SGD(params=model.parameters() ,lr=init_lr , weight_decay=1e-5, momentum=0.9, nesterov=True) 
        lr_scheduler = StepLRScheduler(optimizer, lr_steps, lr_mults, last_iter=-1) 
    # Adam 
    elif optim_type == 'Adam':
        optimizer = optim.Adam(params=model.parameters(), lr=0.01)
        lr_scheduler = None

    dir_name = os.path.join('./logs', name)
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)

    with open('%s/opts.json' % dir_name, 'w') as fp:
        json.dump(vars(opt), fp, indent=1)
    
    model = train_model(model, criterion, optimizer, lr_scheduler, num_epochs=120)
