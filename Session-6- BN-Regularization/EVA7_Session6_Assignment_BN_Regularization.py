# -*- coding: utf-8 -*-
"""main.py

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/swathi-ai/Swathi-TSAI/blob/main/S6/main.ipynb
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/My Drive/TSAI/S6')
#%cd /content/drive/My Drive/TSAI/S6

"""#load data"""

torch.manual_seed(1)
batch_size = 128
use_cuda = torch.cuda.is_available()

kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),                                                  
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

from model import *

device = torch.device("cuda" if use_cuda else "cpu")

"""#Model with Group Normalization"""

GN_model_obj = Net('GN').to(device)
summary(GN_model_obj, input_size=(1, 28, 28))

GN_train_losses = []
GN_test_losses = []
GN_train_acc = []
GN_test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    GN_train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    GN_train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    GN_test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    GN_test_acc.append(100. * correct / len(test_loader.dataset))

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(GN_model_obj.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


EPOCHS = 20
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(GN_model_obj, device, train_loader, optimizer, epoch)
    scheduler.step()
    test(GN_model_obj, device, test_loader)

"""#Model with Layer Normalization"""

LN_model_obj = Net('LN').to(device)
summary(LN_model_obj, input_size=(1, 28, 28))

LN_train_losses = []
LN_test_losses = []
LN_train_acc = []
LN_test_acc = []

def train(model, device, train_loader, optimizer, epoch):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = F.nll_loss(y_pred, target)
    LN_train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    LN_train_acc.append(100*correct/processed)

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    LN_test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    LN_test_acc.append(100. * correct / len(test_loader.dataset))

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(LN_model_obj.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


EPOCHS = 20
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(LN_model_obj, device, train_loader, optimizer, epoch)
    scheduler.step()
    test(LN_model_obj, device, test_loader)

"""#Model with L1 + Batch Normalization"""

L1_BN_model_obj = Net('BN').to(device)
summary(L1_BN_model_obj, input_size=(1, 28, 28))

def l1_penalty(params, l1_lambda=0.001):
  # This is taken from https://stackoverflow.com/questions/44641976/pytorch-how-to-add-l1-regularizer-to-activations
    """Returns the L1 penalty of the params."""
    l1_norm = sum(p.abs().sum() for p in params)
    return l1_lambda*l1_norm

# loss = loss_fn(outputs, labels) + l1_penalty(my_layer.parameters())

L1_BN_train_losses = []
L1_BN_test_losses = []
L1_BN_train_acc = []
L1_BN_test_acc = []

def train(model, device, train_loader, optimizer, epoch):
    correct = 0
    processed = 1
    model.train()
    pbar = tqdm(train_loader)
    ## l1 regularization https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
    # l1_lambda = 0.001
    # l1_norm = sum(p.abs().sum() for p in model.parameters())
    
    for batch_idx, (data, target) in enumerate(pbar):
        # print(len(data))
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        loss = loss + l1_penalty(model.parameters())
        loss.backward()
        L1_BN_train_losses.append(loss)
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        # print(correct)
        processed += len(data)
        L1_BN_train_acc.append(100*correct/processed)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    L1_BN_test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    L1_BN_test_acc.append(100. * correct / len(test_loader.dataset))

#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer = optim.Adam(L1_BN_model_obj.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)


EPOCHS = 20
for epoch in range(EPOCHS):
    print("EPOCH:", epoch)
    train(L1_BN_model_obj, device, train_loader, optimizer, epoch)
    scheduler.step()
    test(L1_BN_model_obj, device, test_loader)

"""#Test/Validation Loss for all 3 models together"""

epochs = range(1,21)
plt.plot(epochs, GN_test_losses, 'g', label='Model with Group Norm')
plt.plot(epochs, LN_test_losses, 'r', label='Model with Layer Norm')
plt.plot(epochs, L1_BN_test_losses, 'b', label='Model with L1 with Batch Norm')

plt.title('Validation loss for different normalization techniques')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""#Test/Validation Accuracy for 3 models together"""

epochs = range(1,21)
plt.plot(epochs, GN_test_acc, 'g', label='Model with Group Norm')
plt.plot(epochs, LN_test_acc, 'r', label='Model with Layer Norm')
plt.plot(epochs, L1_BN_test_acc, 'b', label='Model with L1 with Batch Norm')

plt.title('Validation Acc for different normalization techniques')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""#10 misclassified images for model with Group Normalization"""

def display_misclassified_images(model,
                                test_loader,
                                misclassified_image_row = 2,
                                misclassified_image_col = 5):
    """ This function shows 10 images with their predicted and real labels"""
    incorrect_examples = []
    model.eval()
    with torch.no_grad():
      for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = torch.max(output,1)
        # correct += pred.eq(target.view_as(pred)).sum().item()
        idxs_mask = ((pred == target) == False).nonzero()
        # idxs_mask = (pred - target != 0)
        match = idxs_mask.cpu().numpy()
        
        
        if (idxs_mask.shape[0] > 0 ):
          mydata = list(match.reshape(-1))
          for i in range(len(mydata)-1):
            incorrect_examples.append(data[i].cpu())
            if (len(incorrect_examples) > 10):
               break
            
            # print(f"mydata:{mydata[i]}")
          # print(f"idxs_mask:{idxs_mask}")
          # print(type(match))
          # print(f"idxs_mask reshaped:{type(match.reshape(-1))})")
          # print(f"idxs_mask list:{list(match.reshape(-1))}")
          # print(f"match:{match[0]}, length:{len(match)}")
          # incorrect_examples.append(data[mydata].cpu())
        # if (len(incorrect_examples) > 20):
        #   break ## this is done to ensure we dont store too many images and to save memory
    
   # print(f"Number of misclassified Image:{len(incorrect_examples)}")
    n = 0
    nrows = misclassified_image_row
    ncols = misclassified_image_col
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,)

    for row in range(nrows):
      for col in range(ncols):
        # print(incorrect_examples[n])
        img = incorrect_examples[n]
        # print(img)
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        ax[row,col].imshow(img.reshape(28,28))
        
        # print('\n')
        n += 1

display_misclassified_images(GN_model_obj, test_loader,2, 5)

"""#10 misclassified images for model with Layer Normalization"""

display_misclassified_images(LN_model_obj, test_loader,2, 5)

"""#10 misclassified images for model with L1 + BatchNormalization"""

display_misclassified_images(L1_BN_model_obj, test_loader,2, 5)

