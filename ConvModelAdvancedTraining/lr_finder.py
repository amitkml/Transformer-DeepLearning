import os
import sys
import time
import math
import torch

import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from models import *
from gradcam import *
from utils import *
import os

os.system('pip install torchsummary')
from torchsummary import summary

os.system('pip install torch-lr-finder')

from torch_lr_finder import LRFinder

def lr_finder_ls(start_lr=1e-3, end_lr=0.5, device='cuda', epoch=24):
     
  device = torch.device("cuda")
#   best_acc = 0  # best test accuracy
  start_epoch = 0
  print("Got all parser argument")
  # Data
  print('================================================> Preparing data................')
  
  mean,std = get_mean_and_std()

  train_transforms, test_transforms = data_albumentations_customresnet(mean, std)
  
  trainset = torchvision.datasets.CIFAR10(
  root='./data', train=True, download=True, transform=train_transforms)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=512, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=test_transforms)
  testloader = torch.utils.data.DataLoader(  
      testset, batch_size=512, shuffle=False, num_workers=2)
  

  classes = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')

# Model
  print('===========================================================> Building model Custom resnet...............')
  train_losses = []
  test_losses = []
  train_accuracy = []
  test_accuracy = []
  
# net = VGG('VGG19')
  net = ResNetCustom()
  net = net.to(device)
  
  model_summary(net, device, input_size=(3, 32, 32))
  
  print('/n ================================================================================================== /n')
  print('/n ================================================================================================== /n')    
  exp_metrics={}
  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
  start_lr = start_lr
  end_lr = 0.5
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=start_lr,
                        momentum=0.9, weight_decay=5e-4)
  

  num_iter = len(testloader) * epoch
  print(f"--------------------------------Starting LR Finder Test-----------------------------------------")

  lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
  lr_finder.range_test(trainloader, testloader, start_lr,end_lr, num_iter, step_mode="linear", diverge_th=50)
  max_lr = lr_finder.plot(suggest_lr=True,skip_start=0, skip_end=0)
  # max_lr = lr_finder.history['lr'][lr_finder.history['loss'].index(lr_finder.best_loss)]
  # lr_finder.plot(skip_end=0)
  lr_finder.reset() # to reset the model and optimizer to their initial state
  return max_lr