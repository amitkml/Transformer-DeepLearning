# Assignment

1. Run this [network](https://colab.research.google.com/drive/1qlewMtxcAJT6fIJdmMh8pSf2e-dh51Rw).  
2. Fix the network above:
   1. change the code such that it uses GPU
   2. change the architecture to C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead) **(If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)**
   3. total RF must be more than **52**
   4. **two** of the layers must use Depthwise Separable Convolution
   5. one of the layers must use Dilated Convolution
   6. use GAP (compulsory **mapped to # of classes**):- *CANNOT* add FC after GAP to target #of classes 
   7. use albumentation library and apply:
      1. horizontal flip
      2. shiftScaleRotate 
      3. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
      4. **grayscale**
   8. achieve **87%** accuracy, as many epochs as you want. Total Params to be less than **100k**. 
   9. upload to Github
   10. Attempt S7-Assignment Solution. Questions in the Assignment QnA are:
       1. Which assignment are you submitting? (early/late)
       2. Please mention the name of your partners who are submitting EXACTLY the same assignment. Please note if the assignments are different, then all the names mentioned here will get the lowest score. So please check with your team if they are submitting even a slightly different assignment. 
       3. copy paste your model code from your model.py file (full code) [125]
       4. copy paste output of torchsummary [125]
       5. copy-paste the code where you implemented albumentation transformation for all three transformations [125]
       6. copy paste your training log (you must be running validation/text after each Epoch [125]
       7. Share the link for your README.md file. [200]

# Dataset

CIFAR10 is a collection of images used to train Machine Learning and Computer Vision algorithms. It contains 60K images having dimension of 32x32 with ten different classes such as airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. We train our Neural Net Model specifically Convolutional Neural Net (CNN) on this data set.



# Network Summary

## Model Summary

```
Net(
  (conv1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout2d(p=0.01, inplace=False)
    (4): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (5): ReLU()
    (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout2d(p=0.01, inplace=False)
  )
  (trans1): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2))
    (1): ReLU()
  )
  (conv2): Sequential(
    (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout2d(p=0.01, inplace=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    (5): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
    (6): ReLU()
    (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): Dropout2d(p=0.01, inplace=False)
  )
  (trans2): Sequential(
    (0): Conv2d(64, 32, kernel_size=(1, 1), stride=(2, 2))
    (1): ReLU()
  )
  (conv3): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dilation=(2, 2), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout2d(p=0.01, inplace=False)
    (4): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
    (5): ReLU()
    (6): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout2d(p=0.01, inplace=False)
  )
  (trans3): Sequential(
    (0): Conv2d(64, 16, kernel_size=(1, 1), stride=(2, 2))
    (1): ReLU()
  )
  (conv4): Sequential(
    (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout2d(p=0.01, inplace=False)
    (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
    (5): Conv2d(32, 10, kernel_size=(1, 1), stride=(1, 1), padding=(1, 1), bias=False)
    (6): ReLU()
    (7): BatchNorm2d(10, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (8): Dropout2d(p=0.01, inplace=False)
  )
  (gap): Sequential(
    (0): AdaptiveAvgPool2d(output_size=1)
  )
)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
         Dropout2d-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
         Dropout2d-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 16, 16]           2,080
             ReLU-10           [-1, 32, 16, 16]               0
           Conv2d-11           [-1, 32, 16, 16]           9,216
             ReLU-12           [-1, 32, 16, 16]               0
      BatchNorm2d-13           [-1, 32, 16, 16]              64
        Dropout2d-14           [-1, 32, 16, 16]               0
           Conv2d-15           [-1, 32, 16, 16]             288
           Conv2d-16           [-1, 64, 18, 18]           2,048
             ReLU-17           [-1, 64, 18, 18]               0
      BatchNorm2d-18           [-1, 64, 18, 18]             128
        Dropout2d-19           [-1, 64, 18, 18]               0
           Conv2d-20             [-1, 32, 9, 9]           2,080
             ReLU-21             [-1, 32, 9, 9]               0
           Conv2d-22             [-1, 64, 7, 7]          18,432
             ReLU-23             [-1, 64, 7, 7]               0
      BatchNorm2d-24             [-1, 64, 7, 7]             128
        Dropout2d-25             [-1, 64, 7, 7]               0
           Conv2d-26             [-1, 64, 7, 7]             576
             ReLU-27             [-1, 64, 7, 7]               0
      BatchNorm2d-28             [-1, 64, 7, 7]             128
        Dropout2d-29             [-1, 64, 7, 7]               0
           Conv2d-30             [-1, 16, 4, 4]           1,040
             ReLU-31             [-1, 16, 4, 4]               0
           Conv2d-32             [-1, 32, 4, 4]           4,608
             ReLU-33             [-1, 32, 4, 4]               0
      BatchNorm2d-34             [-1, 32, 4, 4]              64
        Dropout2d-35             [-1, 32, 4, 4]               0
           Conv2d-36             [-1, 32, 4, 4]             288
           Conv2d-37             [-1, 10, 6, 6]             320
             ReLU-38             [-1, 10, 6, 6]               0
      BatchNorm2d-39             [-1, 10, 6, 6]              20
        Dropout2d-40             [-1, 10, 6, 6]               0
AdaptiveAvgPool2d-41             [-1, 10, 1, 1]               0
================================================================
Total params: 60,996
Trainable params: 60,996
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.34
Params size (MB): 0.23
Estimated Total Size (MB): 4.58
```

## Image Augmentation

```
def data_albumentations(horizontalflip_prob = 0.2,
                        rotate_limit = 15,
                        shiftscalerotate_prob = 0.25,
                        num_holes = 1,
                        cutout_prob = 0.5):
    # Calculate mean and std deviation for cifar dataset
    mean,std = helper_functions.calculate_mean_std()
    
    # Train Phase transformations
    train_transforms = A.Compose([A.HorizontalFlip(p=horizontalflip_prob),
                                  A.GaussNoise(p=0.1),
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
```

The above augmentation is being used from cifar10 loader.

```


class Cifar10DataLoader:
    def __init__(self,batchsize=256):
      self.batch_size = batchsize
      self.num_workers = 2
      self.pin_memory = True

      train_transforms, test_transforms = augmentation_albumentations.data_albumentations()

      trainset = datasets.CIFAR10(root='./data', train=True,
                                  download=True, transform=train_transforms)
      
      testset  = datasets.CIFAR10(root='./data', train=False,
                                  download=True, transform=test_transforms)
      
      self.train_loader = torch.utils.data.DataLoader(trainset,
                                                      batch_size = self.batch_size,
                                                      shuffle = True,
                                                      num_workers = self.num_workers,
                                                      pin_memory = self.pin_memory)
      
      self.test_loader = torch.utils.data.DataLoader(testset, 
                                                      batch_size = self.batch_size,
                                                      shuffle = True,
                                                      num_workers = self.num_workers,
                                                      pin_memory = self.pin_memory)


```

## Network Performance

```


Epoch 1:

Loss=1.8337123394012451 Batch_id=195 LR=0.00043 Accuracy=24.07: 100%|██████████| 196/196 [00:26<00:00,  7.48it/s]

Test set: Average loss: 0.0071, Accuracy: 3899/10000 (38.99%)

Epoch 2:

Loss=1.7042181491851807 Batch_id=195 LR=0.00052 Accuracy=37.80: 100%|██████████| 196/196 [00:26<00:00,  7.37it/s]

Test set: Average loss: 0.0063, Accuracy: 4781/10000 (47.81%)

Epoch 3:

Loss=1.733180284500122 Batch_id=195 LR=0.00066 Accuracy=44.39: 100%|██████████| 196/196 [00:26<00:00,  7.42it/s]

Test set: Average loss: 0.0056, Accuracy: 5480/10000 (54.80%)

Epoch 4:

Loss=1.7437212467193604 Batch_id=195 LR=0.00086 Accuracy=49.73: 100%|██████████| 196/196 [00:26<00:00,  7.45it/s]

Test set: Average loss: 0.0051, Accuracy: 5872/10000 (58.72%)

Epoch 5:

Loss=1.3798701763153076 Batch_id=195 LR=0.00111 Accuracy=52.74: 100%|██████████| 196/196 [00:26<00:00,  7.37it/s]

Test set: Average loss: 0.0048, Accuracy: 6020/10000 (60.20%)

Epoch 6:

Loss=1.315139889717102 Batch_id=195 LR=0.00141 Accuracy=55.95: 100%|██████████| 196/196 [00:26<00:00,  7.44it/s]

Test set: Average loss: 0.0044, Accuracy: 6342/10000 (63.42%)

Epoch 7:

Loss=1.2820134162902832 Batch_id=195 LR=0.00176 Accuracy=58.87: 100%|██████████| 196/196 [00:26<00:00,  7.38it/s]

Test set: Average loss: 0.0039, Accuracy: 6780/10000 (67.80%)

Epoch 8:

Loss=1.3094491958618164 Batch_id=195 LR=0.00215 Accuracy=60.54: 100%|██████████| 196/196 [00:26<00:00,  7.48it/s]

Test set: Average loss: 0.0038, Accuracy: 6893/10000 (68.93%)

Epoch 9:

Loss=1.0295288562774658 Batch_id=195 LR=0.00258 Accuracy=62.30: 100%|██████████| 196/196 [00:26<00:00,  7.45it/s]

Test set: Average loss: 0.0038, Accuracy: 6891/10000 (68.91%)

Epoch 10:

Loss=0.8397862315177917 Batch_id=195 LR=0.00303 Accuracy=63.53: 100%|██████████| 196/196 [00:26<00:00,  7.44it/s]

Test set: Average loss: 0.0041, Accuracy: 6598/10000 (65.98%)

Epoch 11:

Loss=1.006030797958374 Batch_id=195 LR=0.00352 Accuracy=64.75: 100%|██████████| 196/196 [00:26<00:00,  7.41it/s]

Test set: Average loss: 0.0037, Accuracy: 6849/10000 (68.49%)

Epoch 12:

Loss=0.8146241903305054 Batch_id=195 LR=0.00402 Accuracy=65.40: 100%|██████████| 196/196 [00:26<00:00,  7.40it/s]

Test set: Average loss: 0.0031, Accuracy: 7429/10000 (74.29%)

Epoch 13:

Loss=0.9492785334587097 Batch_id=195 LR=0.00454 Accuracy=66.17: 100%|██████████| 196/196 [00:26<00:00,  7.42it/s]

Test set: Average loss: 0.0031, Accuracy: 7322/10000 (73.22%)

Epoch 14:

Loss=1.1137462854385376 Batch_id=195 LR=0.00507 Accuracy=66.39: 100%|██████████| 196/196 [00:26<00:00,  7.45it/s]

Test set: Average loss: 0.0031, Accuracy: 7403/10000 (74.03%)

Epoch 15:

Loss=0.8237492442131042 Batch_id=195 LR=0.00560 Accuracy=67.31: 100%|██████████| 196/196 [00:26<00:00,  7.43it/s]

Test set: Average loss: 0.0031, Accuracy: 7355/10000 (73.55%)

Epoch 16:

Loss=1.1166660785675049 Batch_id=195 LR=0.00612 Accuracy=67.79: 100%|██████████| 196/196 [00:26<00:00,  7.41it/s]

Test set: Average loss: 0.0029, Accuracy: 7552/10000 (75.52%)

Epoch 17:

Loss=1.0921255350112915 Batch_id=195 LR=0.00663 Accuracy=67.86: 100%|██████████| 196/196 [00:26<00:00,  7.44it/s]

Test set: Average loss: 0.0029, Accuracy: 7514/10000 (75.14%)

Epoch 18:

Loss=0.8593747019767761 Batch_id=195 LR=0.00713 Accuracy=68.39: 100%|██████████| 196/196 [00:26<00:00,  7.43it/s]

Test set: Average loss: 0.0030, Accuracy: 7374/10000 (73.74%)

Epoch 19:

Loss=0.9844444394111633 Batch_id=195 LR=0.00760 Accuracy=69.19: 100%|██████████| 196/196 [00:26<00:00,  7.40it/s]

Test set: Average loss: 0.0032, Accuracy: 7281/10000 (72.81%)

Epoch 20:

Loss=1.0915096998214722 Batch_id=195 LR=0.00804 Accuracy=68.93: 100%|██████████| 196/196 [00:26<00:00,  7.44it/s]

Test set: Average loss: 0.0029, Accuracy: 7475/10000 (74.75%)

Epoch 21:

Loss=0.8692792654037476 Batch_id=195 LR=0.00845 Accuracy=69.29: 100%|██████████| 196/196 [00:26<00:00,  7.38it/s]

Test set: Average loss: 0.0028, Accuracy: 7627/10000 (76.27%)

Epoch 22:

Loss=0.955003559589386 Batch_id=195 LR=0.00882 Accuracy=69.45: 100%|██████████| 196/196 [00:26<00:00,  7.46it/s]

Test set: Average loss: 0.0029, Accuracy: 7489/10000 (74.89%)

Epoch 23:

Loss=0.8477927446365356 Batch_id=195 LR=0.00915 Accuracy=69.51: 100%|██████████| 196/196 [00:26<00:00,  7.42it/s]

Test set: Average loss: 0.0027, Accuracy: 7681/10000 (76.81%)

Epoch 24:

Loss=0.7892859578132629 Batch_id=195 LR=0.00942 Accuracy=69.71: 100%|██████████| 196/196 [00:26<00:00,  7.44it/s]

Test set: Average loss: 0.0029, Accuracy: 7493/10000 (74.93%)

Epoch 25:

Loss=0.8939248919487 Batch_id=195 LR=0.00965 Accuracy=69.97: 100%|██████████| 196/196 [00:26<00:00,  7.28it/s]

Test set: Average loss: 0.0033, Accuracy: 7243/10000 (72.43%)

Epoch 26:

Loss=0.8531785011291504 Batch_id=195 LR=0.00982 Accuracy=70.07: 100%|██████████| 196/196 [00:26<00:00,  7.45it/s]

Test set: Average loss: 0.0026, Accuracy: 7786/10000 (77.86%)

Epoch 27:

Loss=1.0856314897537231 Batch_id=195 LR=0.00993 Accuracy=70.64: 100%|██████████| 196/196 [00:26<00:00,  7.42it/s]

Test set: Average loss: 0.0036, Accuracy: 7044/10000 (70.44%)

Epoch 28:

Loss=1.1456373929977417 Batch_id=195 LR=0.00999 Accuracy=70.34: 100%|██████████| 196/196 [00:26<00:00,  7.44it/s]

Test set: Average loss: 0.0034, Accuracy: 7208/10000 (72.08%)

Epoch 29:

Loss=1.075331687927246 Batch_id=195 LR=0.01000 Accuracy=70.37: 100%|██████████| 196/196 [00:26<00:00,  7.37it/s]

Test set: Average loss: 0.0030, Accuracy: 7476/10000 (74.76%)

Epoch 30:

Loss=0.9881223440170288 Batch_id=195 LR=0.00999 Accuracy=70.38: 100%|██████████| 196/196 [00:26<00:00,  7.38it/s]

Test set: Average loss: 0.0029, Accuracy: 7618/10000 (76.18%)

Epoch 31:

Loss=0.9886752963066101 Batch_id=195 LR=0.00997 Accuracy=70.76: 100%|██████████| 196/196 [00:26<00:00,  7.34it/s]

Test set: Average loss: 0.0027, Accuracy: 7615/10000 (76.15%)

Epoch 32:

Loss=0.9022455215454102 Batch_id=195 LR=0.00993 Accuracy=70.93: 100%|██████████| 196/196 [00:26<00:00,  7.32it/s]

Test set: Average loss: 0.0028, Accuracy: 7552/10000 (75.52%)

Epoch 33:

Loss=0.9166707992553711 Batch_id=195 LR=0.00989 Accuracy=71.10: 100%|██████████| 196/196 [00:26<00:00,  7.38it/s]

Test set: Average loss: 0.0025, Accuracy: 7853/10000 (78.53%)

Epoch 34:

Loss=0.8106001615524292 Batch_id=195 LR=0.00983 Accuracy=71.00: 100%|██████████| 196/196 [00:26<00:00,  7.34it/s]

Test set: Average loss: 0.0029, Accuracy: 7624/10000 (76.24%)

Epoch 35:

Loss=0.8161551356315613 Batch_id=195 LR=0.00977 Accuracy=71.05: 100%|██████████| 196/196 [00:26<00:00,  7.38it/s]

Test set: Average loss: 0.0033, Accuracy: 7273/10000 (72.73%)

Epoch 36:

Loss=0.8689115643501282 Batch_id=195 LR=0.00969 Accuracy=71.04: 100%|██████████| 196/196 [00:26<00:00,  7.47it/s]

Test set: Average loss: 0.0028, Accuracy: 7608/10000 (76.08%)

Epoch 37:

Loss=0.732958197593689 Batch_id=195 LR=0.00960 Accuracy=71.34: 100%|██████████| 196/196 [00:26<00:00,  7.38it/s]

Test set: Average loss: 0.0025, Accuracy: 7884/10000 (78.84%)

Epoch 38:

Loss=0.8037557601928711 Batch_id=195 LR=0.00950 Accuracy=71.22: 100%|██████████| 196/196 [00:26<00:00,  7.47it/s]

Test set: Average loss: 0.0025, Accuracy: 7805/10000 (78.05%)

Epoch 39:

Loss=0.9890052676200867 Batch_id=195 LR=0.00940 Accuracy=71.57: 100%|██████████| 196/196 [00:26<00:00,  7.42it/s]

Test set: Average loss: 0.0030, Accuracy: 7418/10000 (74.18%)

Epoch 40:

Loss=0.7085970640182495 Batch_id=195 LR=0.00928 Accuracy=71.61: 100%|██████████| 196/196 [00:26<00:00,  7.40it/s]

Test set: Average loss: 0.0026, Accuracy: 7795/10000 (77.95%)

Epoch 41:

Loss=0.6468234062194824 Batch_id=195 LR=0.00915 Accuracy=71.58: 100%|██████████| 196/196 [00:26<00:00,  7.36it/s]

Test set: Average loss: 0.0024, Accuracy: 7932/10000 (79.32%)

Epoch 42:

Loss=0.8490002751350403 Batch_id=195 LR=0.00902 Accuracy=71.77: 100%|██████████| 196/196 [00:26<00:00,  7.41it/s]

Test set: Average loss: 0.0027, Accuracy: 7621/10000 (76.21%)

Epoch 43:

Loss=0.9128476977348328 Batch_id=195 LR=0.00887 Accuracy=71.97: 100%|██████████| 196/196 [00:26<00:00,  7.35it/s]

Test set: Average loss: 0.0027, Accuracy: 7706/10000 (77.06%)

Epoch 44:

Loss=0.8178725242614746 Batch_id=195 LR=0.00872 Accuracy=72.28: 100%|██████████| 196/196 [00:26<00:00,  7.34it/s]

Test set: Average loss: 0.0024, Accuracy: 7962/10000 (79.62%)

Epoch 45:

Loss=0.8195681571960449 Batch_id=195 LR=0.00856 Accuracy=72.10: 100%|██████████| 196/196 [00:26<00:00,  7.45it/s]

Test set: Average loss: 0.0026, Accuracy: 7828/10000 (78.28%)

Epoch 46:

Loss=0.6943470239639282 Batch_id=195 LR=0.00839 Accuracy=72.61: 100%|██████████| 196/196 [00:26<00:00,  7.37it/s]

Test set: Average loss: 0.0024, Accuracy: 7977/10000 (79.77%)

Epoch 47:

Loss=0.8309850692749023 Batch_id=195 LR=0.00821 Accuracy=72.30: 100%|██████████| 196/196 [00:26<00:00,  7.29it/s]

Test set: Average loss: 0.0027, Accuracy: 7692/10000 (76.92%)

Epoch 48:

Loss=0.6766585111618042 Batch_id=195 LR=0.00802 Accuracy=72.46: 100%|██████████| 196/196 [00:26<00:00,  7.36it/s]

Test set: Average loss: 0.0027, Accuracy: 7752/10000 (77.52%)

Epoch 49:

Loss=1.0532644987106323 Batch_id=195 LR=0.00783 Accuracy=73.02: 100%|██████████| 196/196 [00:26<00:00,  7.31it/s]

Test set: Average loss: 0.0027, Accuracy: 7721/10000 (77.21%)

Epoch 50:

Loss=0.8297349810600281 Batch_id=195 LR=0.00763 Accuracy=72.99: 100%|██████████| 196/196 [00:26<00:00,  7.31it/s]

Test set: Average loss: 0.0025, Accuracy: 7892/10000 (78.92%)

Epoch 51:

Loss=0.8353104591369629 Batch_id=195 LR=0.00743 Accuracy=73.20: 100%|██████████| 196/196 [00:26<00:00,  7.35it/s]

Test set: Average loss: 0.0023, Accuracy: 8027/10000 (80.27%)

Epoch 52:

Loss=0.9741862416267395 Batch_id=195 LR=0.00722 Accuracy=73.42: 100%|██████████| 196/196 [00:26<00:00,  7.36it/s]

Test set: Average loss: 0.0024, Accuracy: 7930/10000 (79.30%)

Epoch 53:

Loss=0.5614804625511169 Batch_id=195 LR=0.00701 Accuracy=73.45: 100%|██████████| 196/196 [00:26<00:00,  7.41it/s]

Test set: Average loss: 0.0024, Accuracy: 7947/10000 (79.47%)

Epoch 54:

Loss=0.8011579513549805 Batch_id=195 LR=0.00679 Accuracy=74.02: 100%|██████████| 196/196 [00:26<00:00,  7.34it/s]

Test set: Average loss: 0.0025, Accuracy: 7923/10000 (79.23%)

Epoch 55:

Loss=0.7982090711593628 Batch_id=195 LR=0.00657 Accuracy=73.97: 100%|██████████| 196/196 [00:26<00:00,  7.37it/s]

Test set: Average loss: 0.0024, Accuracy: 7999/10000 (79.99%)

Epoch 56:

Loss=0.77245032787323 Batch_id=195 LR=0.00634 Accuracy=73.90: 100%|██████████| 196/196 [00:26<00:00,  7.33it/s]

Test set: Average loss: 0.0022, Accuracy: 8166/10000 (81.66%)

Epoch 57:

Loss=0.7591160535812378 Batch_id=195 LR=0.00611 Accuracy=74.28: 100%|██████████| 196/196 [00:26<00:00,  7.35it/s]

Test set: Average loss: 0.0023, Accuracy: 8041/10000 (80.41%)

Epoch 58:

Loss=0.6082742810249329 Batch_id=195 LR=0.00588 Accuracy=74.43: 100%|██████████| 196/196 [00:26<00:00,  7.43it/s]

Test set: Average loss: 0.0023, Accuracy: 8031/10000 (80.31%)

Epoch 59:

Loss=0.77958083152771 Batch_id=195 LR=0.00565 Accuracy=74.65: 100%|██████████| 196/196 [00:26<00:00,  7.33it/s]

Test set: Average loss: 0.0022, Accuracy: 8144/10000 (81.44%)

Epoch 60:

Loss=0.885550856590271 Batch_id=195 LR=0.00541 Accuracy=75.05: 100%|██████████| 196/196 [00:26<00:00,  7.39it/s]

Test set: Average loss: 0.0022, Accuracy: 8085/10000 (80.85%)

Epoch 61:

Loss=0.7457245588302612 Batch_id=195 LR=0.00518 Accuracy=75.09: 100%|██████████| 196/196 [00:26<00:00,  7.32it/s]

Test set: Average loss: 0.0023, Accuracy: 8101/10000 (81.01%)

Epoch 62:

Loss=0.856988787651062 Batch_id=195 LR=0.00494 Accuracy=75.52: 100%|██████████| 196/196 [00:26<00:00,  7.39it/s]

Test set: Average loss: 0.0021, Accuracy: 8220/10000 (82.20%)

Epoch 63:

Loss=0.6839286088943481 Batch_id=195 LR=0.00470 Accuracy=75.79: 100%|██████████| 196/196 [00:26<00:00,  7.35it/s]

Test set: Average loss: 0.0021, Accuracy: 8148/10000 (81.48%)

Epoch 64:

Loss=0.5672301054000854 Batch_id=195 LR=0.00447 Accuracy=75.67: 100%|██████████| 196/196 [00:27<00:00,  7.25it/s]

Test set: Average loss: 0.0020, Accuracy: 8282/10000 (82.82%)

Epoch 65:

Loss=0.6435562372207642 Batch_id=195 LR=0.00423 Accuracy=75.97: 100%|██████████| 196/196 [00:27<00:00,  7.19it/s]

Test set: Average loss: 0.0021, Accuracy: 8241/10000 (82.41%)

Epoch 66:

Loss=0.5281286239624023 Batch_id=195 LR=0.00400 Accuracy=76.52: 100%|██████████| 196/196 [00:26<00:00,  7.34it/s]

Test set: Average loss: 0.0020, Accuracy: 8303/10000 (83.03%)

Epoch 67:

Loss=0.690117359161377 Batch_id=195 LR=0.00377 Accuracy=76.52: 100%|██████████| 196/196 [00:26<00:00,  7.33it/s]

Test set: Average loss: 0.0020, Accuracy: 8320/10000 (83.20%)

Epoch 68:

Loss=0.9565600156784058 Batch_id=195 LR=0.00354 Accuracy=76.71: 100%|██████████| 196/196 [00:26<00:00,  7.33it/s]

Test set: Average loss: 0.0019, Accuracy: 8381/10000 (83.81%)

Epoch 69:

Loss=0.49816450476646423 Batch_id=195 LR=0.00332 Accuracy=77.30: 100%|██████████| 196/196 [00:26<00:00,  7.31it/s]

Test set: Average loss: 0.0019, Accuracy: 8381/10000 (83.81%)

Epoch 70:

Loss=0.7226431965827942 Batch_id=195 LR=0.00310 Accuracy=77.39: 100%|██████████| 196/196 [00:26<00:00,  7.33it/s]

Test set: Average loss: 0.0019, Accuracy: 8368/10000 (83.68%)

Epoch 71:

Loss=0.6356824636459351 Batch_id=195 LR=0.00288 Accuracy=77.83: 100%|██████████| 196/196 [00:26<00:00,  7.30it/s]

Test set: Average loss: 0.0019, Accuracy: 8365/10000 (83.65%)

Epoch 72:

Loss=0.6859714388847351 Batch_id=195 LR=0.00267 Accuracy=77.94: 100%|██████████| 196/196 [00:26<00:00,  7.29it/s]

Test set: Average loss: 0.0018, Accuracy: 8436/10000 (84.36%)

Epoch 73:

Loss=0.6100456714630127 Batch_id=195 LR=0.00246 Accuracy=78.46: 100%|██████████| 196/196 [00:26<00:00,  7.31it/s]

Test set: Average loss: 0.0018, Accuracy: 8421/10000 (84.21%)

Epoch 74:

Loss=0.6661063432693481 Batch_id=195 LR=0.00226 Accuracy=78.57: 100%|██████████| 196/196 [00:26<00:00,  7.27it/s]

Test set: Average loss: 0.0018, Accuracy: 8495/10000 (84.95%)

Epoch 75:

Loss=0.645595133304596 Batch_id=195 LR=0.00207 Accuracy=78.71: 100%|██████████| 196/196 [00:26<00:00,  7.32it/s]

Test set: Average loss: 0.0018, Accuracy: 8495/10000 (84.95%)

Epoch 76:

Loss=0.7277189493179321 Batch_id=195 LR=0.00188 Accuracy=78.87: 100%|██████████| 196/196 [00:26<00:00,  7.30it/s]

Test set: Average loss: 0.0017, Accuracy: 8514/10000 (85.14%)

Epoch 77:

Loss=0.5693426728248596 Batch_id=195 LR=0.00170 Accuracy=79.25: 100%|██████████| 196/196 [00:26<00:00,  7.29it/s]

Test set: Average loss: 0.0017, Accuracy: 8504/10000 (85.04%)

Epoch 78:

Loss=0.49850964546203613 Batch_id=195 LR=0.00153 Accuracy=79.43: 100%|██████████| 196/196 [00:26<00:00,  7.31it/s]

Test set: Average loss: 0.0017, Accuracy: 8509/10000 (85.09%)

Epoch 79:

Loss=0.6522964239120483 Batch_id=195 LR=0.00136 Accuracy=79.59: 100%|██████████| 196/196 [00:26<00:00,  7.31it/s]

Test set: Average loss: 0.0017, Accuracy: 8567/10000 (85.67%)

Epoch 80:

Loss=0.5523295998573303 Batch_id=195 LR=0.00120 Accuracy=79.81: 100%|██████████| 196/196 [00:26<00:00,  7.28it/s]

Test set: Average loss: 0.0017, Accuracy: 8552/10000 (85.52%)

Epoch 81:

Loss=0.5941554307937622 Batch_id=195 LR=0.00105 Accuracy=80.24: 100%|██████████| 196/196 [00:26<00:00,  7.34it/s]

Test set: Average loss: 0.0017, Accuracy: 8590/10000 (85.90%)

Epoch 82:

Loss=0.6956799626350403 Batch_id=195 LR=0.00091 Accuracy=80.22: 100%|██████████| 196/196 [00:26<00:00,  7.28it/s]

Test set: Average loss: 0.0017, Accuracy: 8584/10000 (85.84%)

Epoch 83:

Loss=0.5770138502120972 Batch_id=195 LR=0.00078 Accuracy=80.60: 100%|██████████| 196/196 [00:27<00:00,  7.24it/s]

Test set: Average loss: 0.0016, Accuracy: 8634/10000 (86.34%)

Epoch 84:

Loss=0.6695526838302612 Batch_id=195 LR=0.00066 Accuracy=80.61: 100%|██████████| 196/196 [00:26<00:00,  7.28it/s]

Test set: Average loss: 0.0017, Accuracy: 8595/10000 (85.95%)

Epoch 85:

Loss=0.48073941469192505 Batch_id=195 LR=0.00055 Accuracy=80.91: 100%|██████████| 196/196 [00:27<00:00,  7.24it/s]

Test set: Average loss: 0.0016, Accuracy: 8624/10000 (86.24%)

Epoch 86:

Loss=0.4539763331413269 Batch_id=195 LR=0.00044 Accuracy=81.02: 100%|██████████| 196/196 [00:26<00:00,  7.26it/s]

Test set: Average loss: 0.0016, Accuracy: 8645/10000 (86.45%)

Epoch 87:

Loss=0.5293794870376587 Batch_id=195 LR=0.00035 Accuracy=81.31: 100%|██████████| 196/196 [00:26<00:00,  7.26it/s]

Test set: Average loss: 0.0016, Accuracy: 8671/10000 (86.71%)

Epoch 88:

Loss=0.5300928950309753 Batch_id=195 LR=0.00027 Accuracy=81.43: 100%|██████████| 196/196 [00:26<00:00,  7.27it/s]

Test set: Average loss: 0.0016, Accuracy: 8679/10000 (86.79%)

Epoch 89:

Loss=0.6882849931716919 Batch_id=195 LR=0.00020 Accuracy=81.60: 100%|██████████| 196/196 [00:27<00:00,  7.23it/s]

Test set: Average loss: 0.0016, Accuracy: 8656/10000 (86.56%)

Epoch 90:

Loss=0.43820643424987793 Batch_id=195 LR=0.00014 Accuracy=81.72: 100%|██████████| 196/196 [00:26<00:00,  7.30it/s]

Test set: Average loss: 0.0016, Accuracy: 8692/10000 (86.92%)

Epoch 91:

Loss=0.5746750235557556 Batch_id=195 LR=0.00009 Accuracy=81.82: 100%|██████████| 196/196 [00:26<00:00,  7.27it/s]

Test set: Average loss: 0.0016, Accuracy: 8675/10000 (86.75%)

Epoch 92:

Loss=0.4375040531158447 Batch_id=195 LR=0.00005 Accuracy=81.54: 100%|██████████| 196/196 [00:27<00:00,  7.23it/s]

Test set: Average loss: 0.0016, Accuracy: 8689/10000 (86.89%)

Epoch 93:

Loss=0.6666744947433472 Batch_id=195 LR=0.00002 Accuracy=81.54: 100%|██████████| 196/196 [00:26<00:00,  7.30it/s]

Test set: Average loss: 0.0015, Accuracy: 8677/10000 (86.77%)

Epoch 94:

Loss=0.3979238271713257 Batch_id=195 LR=0.00001 Accuracy=81.78: 100%|██████████| 196/196 [00:26<00:00,  7.26it/s]

Test set: Average loss: 0.0016, Accuracy: 8688/10000 (86.88%)

Epoch 95:

Loss=0.6644643545150757 Batch_id=195 LR=0.00000 Accuracy=81.92: 100%|██████████| 196/196 [00:27<00:00,  7.26it/s]

Test set: Average loss: 0.0015, Accuracy: 8685/10000 (86.85%)
```

## Learning Rate

Have used one cycle LR as shared below.

```
  l1_factor = 0
  l2_factor = 0.0001
  # optim_type = optim.Adam
  criterion = nn.CrossEntropyLoss()
  # opt_func = optim.Adam
  lr = 0.01
  grad_clip = 0.1
  train_losses = []
  test_losses = []
  train_accuracy = []
  test_accuracy = []
  lrs=[]

  model = model
  optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_factor)
  scheduler = OneCycleLR(optimizer, max_lr=lr,epochs=epochs,steps_per_epoch=len(dataset.train_loader))

  for epoch in range(1, epochs + 1):
    print(f'Epoch {epoch}:')
    
    model_training.train(model, device, dataset.train_loader, optimizer,epoch, train_accuracy, train_losses, l1_factor,scheduler,criterion,lrs,grad_clip)
    model_training.test(model, device, dataset.test_loader,test_accuracy,test_losses,criterion)
```



## Class wise summary

This is quite clear that that model is still mixing up between Dog and Cat.

```
Accuracy of airplane : 93 %
Accuracy of automobile : 94 %
Accuracy of  bird : 92 %
Accuracy of   cat : 63 %
Accuracy of  deer : 91 %
Accuracy of   dog : 84 %
Accuracy of  frog : 90 %
Accuracy of horse : 87 %
Accuracy of  ship : 94 %
Accuracy of truck : 78 %
```

