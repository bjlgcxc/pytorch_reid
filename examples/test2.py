# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import argparse
import os
import sys
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())
import scipy.io
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from models import ResNet50 

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Testing')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='../dataset/Market/pytorch',type=str, help='./test_data')
parser.add_argument('--name', default='ResNet50', type=str, help='save model path')
parser.add_argument('--save', default='logs/pytorch_result.mat', type=str, help='path of test result file')
parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
parser.add_argument('--metric', default=None, type=str, help='metric in [cosface, arcface, sphereface]')
parser.add_argument('--margin', default=None, type=float, help='margin')
parser.add_argument('--scalar', default=None, type=float, help='scalar')
parser.add_argument('--feat_size', default=1024, type=int, help='feature size')

opt = parser.parse_args()

which_epoch = opt.which_epoch
name = opt.name
save = opt.save
test_dir = opt.test_dir
metric = opt.metric
margin = opt.margin
scalar = opt.scalar
feat_size = opt.feat_size

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((288, 144), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                             shuffle=False, num_workers=0) for x in ['gallery','query']}

class_names = image_datasets['query'].classes

def get_id(img_path):    
    camera_id = []
    labels = []
    for path, v in img_path:
        filename = path.split('/')[-1]
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)


# Save to Matlab for check
#result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
#scipy.io.savemat(save, result)

np.save('test_probe_labels.npy', query_label)
np.save('test_gallery_labels.npy', gallery_label)
np.save('test_probe_features.npy', query_feature.numpy())
np.save('test_gallery_features.npy',gallery_feature.numpy() )
