'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
# from utils import progress_bar
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR,OneCycleLR
from models import *
from utils import *
os.system('pip install -U albumentations')

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run_experiments(lr = 0.1, resume = '', description = 'PyTorchCIFAR10Training', epoch =20, lr_scheduler='ReduceLROnPlateau'):
  
 # https://stackoverflow.com/questions/45823991/argparse-in-ipython-notebook-unrecognized-arguments-f
#   parser = argparse.ArgumentParser()
#   parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
  lr = lr
  resume = resume
  epoch = epoch
#   args = parser.parse_args(args=['--lr', lr, '--resume', 'store_true'])
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
#   best_acc = 0  # best test accuracy
  start_epoch = 0
  print("Got all parser argument")
  # Data
  print('====================> Preparing data................')
  
  mean,std = get_mean_and_std()

  train_transforms, test_transforms = data_albumentations(mean, std)
  
  transform_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  trainset = torchvision.datasets.CIFAR10(
  root='./data', train=True, download=True, transform=train_transforms)
  trainloader = torch.utils.data.DataLoader(
      trainset, batch_size=128, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(
      root='./data', train=False, download=True, transform=test_transforms)
  testloader = torch.utils.data.DataLoader(  
      testset, batch_size=100, shuffle=False, num_workers=2)
  

  classes = ('plane', 'car', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck')

# Model
  print('=======================> Building model...............')
  train_losses = []
  test_losses = []
  train_accuracy = []
  test_accuracy = []
  
# net = VGG('VGG19')
  net = ResNet18()
  net = net.to(device)
  
  model_summary(net, device, input_size=(3, 32, 32))
  
  print('/n ======================================================================= /n')    
  exp_metrics={}
  if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True

  if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr,
                        momentum=0.9, weight_decay=5e-4)
  # x = 2 if i > 100 else 1 if i < 100 else 0
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) if lr_scheduler == 'CosineAnnealingLR' else ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True) if  lr_scheduler == 'ReduceLROnPlateau' else OneCycleLR(optimizer, max_lr=lr,epochs=epoch,steps_per_epoch=len(trainloader))
  # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
  for epoch in range(start_epoch, start_epoch+epoch):
      train(epoch, net, optimizer, trainloader, device, criterion, train_losses, train_accuracy)
      test(epoch, net, optimizer, testloader, device, criterion, test_losses, test_accuracy)
      scheduler.step()
  print('/n ============================ Training and Testing Performance ======================== /n')
  exp_metrics[description] = (train_accuracy,train_losses,test_accuracy,test_losses)
  
  print('/n ======================== Class Level Accuracy ============================== /n')
  class_level_accuracy(net, testloader, device)
  
  print('/n ======================== Random Misclassified Images ========================== /n')
  wrong_predictions(testloader, use_cuda, net)
  
# Training
def train(epoch, model, optimizer, trainloader, device, criterion, train_losses, train_accuracy):
    criterion = criterion
    device = device
    trainloader = trainloader
    optimizer = optimizer
    net = model
    print('\nEpoch: %d' % epoch)
    net.train()
    # train_loss = 0
    correct = 0
    total = 0
    processed = 0
    lrs=[]
    
    pbar = tqdm(trainloader)
    
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # train_loss += loss.item()
        train_losses.append(loss.data.cpu().numpy().item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        processed += len(inputs)
        lrs.append(get_lr(optimizer))
        
        # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={get_lr(optimizer):0.5f} Accuracy={100*correct/processed:0.2f}')
        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx}  LR={lrs[-1]:0.5f} Accuracy={100*correct/processed:0.2f}')
        train_accuracy.append(100*correct/processed)
        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, model, optimizer, testloader, device, criterion, test_losses, test_accuracy):
    criterion = criterion
    device = device
    testloader = testloader
    optimizer = optimizer
    net = model
    global best_acc
    net.eval()
    # test_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(testloader)
    lrs=[]
    lrs.append(get_lr(optimizer))
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            # test_loss += loss.item()
            test_losses.append(loss.data.cpu().numpy().item())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={get_lr(optimizer):0.5f} Accuracy={100*correct/total:0.2f}')
            # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={get_lr(optimizer):0.5f} Accuracy={100*correct/total:0.2f}')
            pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]:0.5f} Accuracy={100*correct/total:0.2f}')
            test_accuracy.append(100*correct/total)
            

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc
