'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
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
import os

## This is for gradcam and taken from https://github.com/jacobgil/pytorch-grad-cam
# os.system('pip install grad-cam')
# from pytorch_grad_cam import GradCAM, \
#     ScoreCAM, \
#     GradCAMPlusPlus, \
#     AblationCAM, \
#     XGradCAM, \
#     EigenCAM, \
#     EigenGradCAM, \
#     LayerCAM, \
#     FullGrad
# from pytorch_grad_cam.utils.image import show_cam_on_image, \
#     preprocess_image

## this is for displayig model summary
os.system('pip install torchsummary')
from torchsummary import summary

## this is for image augmentation and taken from albumentations
os.system('pip install -U albumentations')
from torchvision import datasets
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_grad_cam(model, input_tensor, rgb_img, device):
    target_layers = [model.layer4[-1]]
    input_tensor = input_tensor # Create an input tensor image for your model..
    # Construct the CAM object once, and then re-use it on many images:
    
    if device == 'cuda':
        use_cuda = True
    else:
        use_cuda = False
        
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=use_cuda)
    target_category = 281
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
# EXPERIMENT horizontalflip_prob increased to 0.3 from 0.2, rotate_limit to 20 from 15
def data_albumentations(mean,std, horizontalflip_prob = 0.2,
                        rotate_limit = 15,
                        shiftscalerotate_prob = 0.25,
                        num_holes = 1,
                        random_crop_p = 0.2,
                        cutout_prob = 0.5):
    # Calculate mean and std deviation for cifar dataset
    
    
    # Train Phase transformations
    train_transforms = A.Compose([A.HorizontalFlip(p=horizontalflip_prob),
                                  A.RandomCrop(height=4, width=4, p=random_crop_p),
                                  A.PadIfNeeded(32, 32, p=1.0, value=tuple([x * 255.0 for x in mean])),
                                  A.GaussNoise(p=0.1),
                                #   A.cutout(num_hole s=num_holes, max_h_size=16, max_w_size=16, p=cutout_prob),
                                  A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=rotate_limit, p=shiftscalerotate_prob),
                                  A.CoarseDropout(max_holes=num_holes,min_holes = 1, max_height=16, max_width=16, 
                                  p=cutout_prob,fill_value=tuple([x * 255.0 for x in mean]),
                                  min_height=16, min_width=16),
                                  A.ColorJitter(p=0.25,brightness=0.3, contrast=0.3, saturation=0.30, hue=0.2),
                                  A.ToGray(p=0.2),
                                  A.Normalize(mean=mean, std=std,always_apply=True),
                                  ToTensorV2()
                                ])

    # Test Phase transformations
    test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])

    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

# pip install torchsummary

def get_mean_and_std():
    '''Compute the mean and std value of dataset.'''
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    mean = train_set.data.mean(axis=(0,1,2))/255
    std = train_set.data.std(axis=(0,1,2))/255
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)
## This is handling the case where the number of parameters is too few as i am running from colab
# try:
#     _, term_width = os.popen('stty size', 'r').read().split()
#     term_width = 150
# except:
#     term_width = 150

term_width = 150
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
# def progress_bar(current, total, msg=None):
#     global last_time, begin_time
#     if current == 0:
#         begin_time = time.time()  # Reset for new bar.

#     cur_len = int(TOTAL_BAR_LENGTH*current/total)
#     rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

#     sys.stdout.write(' [')
#     for i in range(cur_len):
#         sys.stdout.write('=')
#     sys.stdout.write('>')
#     for i in range(rest_len):
#         sys.stdout.write('.')
#     sys.stdout.write(']')

#     cur_time = time.time()
#     step_time = cur_time - last_time
#     last_time = cur_time
#     tot_time = cur_time - begin_time

#     L = []
#     L.append('  Step: %s' % format_time(step_time))
#     L.append(' | Tot: %s' % format_time(tot_time))
#     if msg:
#         L.append(' | ' + msg)

#     msg = ''.join(L)
#     sys.stdout.write(msg)
#     for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
#         sys.stdout.write(' ')

#     # Go back to the center of the bar.
#     for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
#         sys.stdout.write('\b')
#     sys.stdout.write(' %d/%d ' % (current+1, total))

#     if current < total-1:
#         sys.stdout.write('\r')
#     else:
#         sys.stdout.write('\n')
#     sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def plot_metrics(exp_metrics):
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25,6)
    train_accuracy,train_losses,test_accuracy,test_losses  = exp_metrics
    
    # Plot the learning curve.
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.plot(np.array(test_losses), 'b', label="Validation Loss")
    
    # Label the plot.
    ax1.set_title("Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    
    ax2.plot(np.array(test_accuracy), 'b', label="Validation Accuracy")
    
    # Label the plot.
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.show()
    
def class_level_accuracy(model, test_loader, device, 
                         class_correct = list(0. for i in range(10)),
                         class_total = list(0. for i in range(10))):
  # specify the image classes
  classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
             'dog', 'frog', 'horse', 'ship', 'truck']
  with torch.no_grad():
    for data in test_loader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
  for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

def wrong_predictions(test_loader,
                      use_cuda,
                      model):
        class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        wrong_images=[]
        wrong_label=[]
        correct_label=[]
        with torch.no_grad():
            for data, target in test_loader:
                if use_cuda:
                  data = data.cuda()
                  target = target.cuda()

                output = model(data)        
                pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
                wrong_pred = (pred.eq(target.view_as(pred)) == False)
                wrong_images.append(data[wrong_pred])
                wrong_label.append(pred[wrong_pred])
                correct_label.append(target.view_as(pred)[wrong_pred])  
      
                wrong_predictions = list(zip(torch.cat(wrong_images),torch.cat(wrong_label),torch.cat(correct_label)))    
            print(f'Total wrong predictions are {len(wrong_predictions)}')
      
      
            fig = plt.figure(figsize=(18,20))
            fig.tight_layout()
            # mean,std = helper.calculate_mean_std("CIFAR10")
            for i, (img, pred, correct) in enumerate(wrong_predictions[:10]):
                  img, pred, target = img.cpu().numpy(), pred.cpu(), correct.cpu()
        
                  #mean = torch.FloatTensor(mean).view( 3, 1, 1).expand_as(img).cpu()
                  #std = torch.FloatTensor(std).view( 3, 1, 1).expand_as(img).cpu()
                  #img = img.mul(std).add(mean)
                  #img=img.numpy()
                  
                  img = np.transpose(img, (1, 2, 0)) / 2 + 0.5
                  ax = fig.add_subplot(5, 5, i+1)
                  ax.axis('off')
                  ax.set_title(f'\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}',fontsize=10)  
                  ax.imshow(img)  
          
            plt.show()
        return wrong_predictions
            
def model_summary(model, device, input_size=(3, 32, 32)):
    print(model)
    summary(model,input_size)

def unnormalize(img):
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
#   mean,std = calculate_mean_std("CIFAR")
    img = img.cpu().numpy().astype(dtype=np.float32)
  
    for i in range(img.shape[0]):
        img[i] = (img[i]*std[i])+mean[i]
  
    return np.transpose(img, (1,2,0))

def data_albumentations_customresnet(mean,std, horizontalflip_prob = 0.2,
                        rotate_limit = 15,
                        shiftscalerotate_prob = 0.25,
                        num_holes = 2,
                        random_crop_p = 0.2,
                        cutout_prob = 0.8,
                        PAD=4):


    train_transforms = A.Compose(
        [
         # RandomCrop with Padding
         A.Sequential(
             [
              A.PadIfNeeded(min_height=32+PAD, min_width=32+PAD, always_apply=True),
              A.RandomCrop(width=32, height=32, p=1),
              ],
              p=1,
              ),
         # Horizontal Flipping
         A.HorizontalFlip(p=0.5),
         # Cutout
         A.CoarseDropout(
             max_holes=1,
             max_height=8,
             max_width=8,
             min_holes=1,
             min_height=8,
             min_width=8,
             fill_value=tuple((x * 255.0 for x in mean)),
             p=0.5,
             ),
            A.Normalize(mean=mean, std=std, always_apply=True),
         ToTensorV2(),
        ]
    )        
    # Test Phase transformations
    test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])

    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

def data_albumentations_customresnet_fc(mean,std, max_holes=4,horizontalflip_prob = 0.2,
                        rotate_limit = 15,
                        shiftscalerotate_prob = 0.25,
                        random_crop_p = 0.2,
                        cutout_prob = 0.8,
                        PAD=4):


    train_transforms = A.Compose(
        [
         # RandomCrop with Padding
         A.Sequential(
             [
              A.PadIfNeeded(min_height=32+PAD, min_width=32+PAD, always_apply=True),
              A.RandomCrop(width=32, height=32, p=1),
              ],
              p=1,
              ),
         # Horizontal Flipping
         A.HorizontalFlip(p=0.5),
         # Cutout
         A.CoarseDropout(
             max_holes=max_holes,
             max_height=8,
             max_width=8,
             min_holes=1,
             min_height=8,
             min_width=8,
             fill_value=tuple((x * 255.0 for x in mean)),
             p=0.5,
             ),
            A.Normalize(mean=mean, std=std, always_apply=True),
         ToTensorV2(),
        ]
    )        
    # Test Phase transformations
    test_transforms = A.Compose([A.Normalize(mean=mean, std=std, always_apply=True),
                                 ToTensorV2()])

    return lambda img:train_transforms(image=np.array(img))["image"],lambda img:test_transforms(image=np.array(img))["image"]

def has_cuda():
    	return torch.cuda.is_available()

def which_device():
	return torch.device("cuda" if has_cuda() else "cpu")

def init_seed(args):
	torch.manual_seed(args.seed)

	if has_cuda():
		print("CUDA Available")
		torch.cuda.manual_seed(args.seed)

def show_model_summary(model, input_size):
	print(summary(model, input_size=input_size))

def imshow(img, title, numpy_conversion=True):
	img = unnormalize(img)
    if numpy_conversion:
            img = img.numpy()
    else:
        pass
	fig = plt.figure(figsize=(15,7))
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.title(title)