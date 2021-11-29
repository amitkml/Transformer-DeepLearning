# Assignment

- Write a custom ResNet architecture for CIFAR10 that has the following architecture:
  - PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
  - Layer1 -
    - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    - R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    - Add(X, R1)
  - Layer 2 -
    - Conv 3x3 [256k]
    - MaxPooling2D
    - BN
    - ReLU
  - Layer 3 -
    - X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    - R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
  - Add(X, R2)
  - MaxPooling with Kernel Size 4
  - FC Layer 
  - SoftMax
- Uses One Cycle Policy such that:
- Total Epochs = 24
- Max at Epoch = 5
- LRMIN = FIND
- LRMAX = FIND
- NO Annihilation
- Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
- Batch size = 512
- Target Accuracy: 90% (93% for late submission or double scores). 
NO score if your code is not modular. Your collab must be importing your GitHub package, and then just running the model. I should be able to find the custom_resnet.py model in your GitHub repo that you'd be training. 

# Solution Details

## Model Summary

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
            Conv2d-4          [-1, 128, 32, 32]          73,728
         MaxPool2d-5          [-1, 128, 16, 16]               0
       BatchNorm2d-6          [-1, 128, 16, 16]             256
              ReLU-7          [-1, 128, 16, 16]               0
         Dropout2d-8          [-1, 128, 16, 16]               0
            Conv2d-9          [-1, 128, 16, 16]         147,456
      BatchNorm2d-10          [-1, 128, 16, 16]             256
             ReLU-11          [-1, 128, 16, 16]               0
           Conv2d-12          [-1, 128, 16, 16]         147,456
      BatchNorm2d-13          [-1, 128, 16, 16]             256
             ReLU-14          [-1, 128, 16, 16]               0
       BasicBlock-15          [-1, 128, 16, 16]               0
           Conv2d-16          [-1, 256, 16, 16]         294,912
        MaxPool2d-17            [-1, 256, 8, 8]               0
      BatchNorm2d-18            [-1, 256, 8, 8]             512
             ReLU-19            [-1, 256, 8, 8]               0
        Dropout2d-20            [-1, 256, 8, 8]               0
           Conv2d-21            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-22            [-1, 512, 4, 4]               0
      BatchNorm2d-23            [-1, 512, 4, 4]           1,024
             ReLU-24            [-1, 512, 4, 4]               0
        Dropout2d-25            [-1, 512, 4, 4]               0
           Conv2d-26            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-27            [-1, 512, 4, 4]           1,024
             ReLU-28            [-1, 512, 4, 4]               0
           Conv2d-29            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-30            [-1, 512, 4, 4]           1,024
             ReLU-31            [-1, 512, 4, 4]               0
       BasicBlock-32            [-1, 512, 4, 4]               0
        MaxPool2d-33            [-1, 512, 1, 1]               0
           Linear-34                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 7.19
Params size (MB): 25.07
Estimated Total Size (MB): 32.28
```
## Model Training and Test Result

```
Adjusting learning rate of group 0 to 6.0909e-04.

Epoch: 0

Loss=1.0406466722488403 Batch_id=97  LR=0.00061 Accuracy=48.67: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=1.2786208391189575 Batch_id=19 LR=0.00061 Accuracy=58.68: 100%|██████████| 20/20 [00:04<00:00,  4.85it/s]

Adjusting learning rate of group 0 to 6.0927e-04.

Epoch: 1

Loss=0.7833951115608215 Batch_id=97  LR=0.00061 Accuracy=67.49: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.7709842324256897 Batch_id=19 LR=0.00061 Accuracy=74.94: 100%|██████████| 20/20 [00:04<00:00,  4.88it/s]

Adjusting learning rate of group 0 to 6.0982e-04.

Epoch: 2

Loss=0.6780174374580383 Batch_id=97  LR=0.00061 Accuracy=75.66: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.6064265370368958 Batch_id=19 LR=0.00061 Accuracy=78.81: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.1073e-04.

Epoch: 3

Loss=0.6664482951164246 Batch_id=97  LR=0.00061 Accuracy=79.74: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.5236298441886902 Batch_id=19 LR=0.00061 Accuracy=82.19: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.1201e-04.

Epoch: 4

Loss=0.5057963728904724 Batch_id=97  LR=0.00061 Accuracy=82.37: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.47396206855773926 Batch_id=19 LR=0.00061 Accuracy=83.95: 100%|██████████| 20/20 [00:04<00:00,  4.87it/s]

Adjusting learning rate of group 0 to 6.1365e-04.

Epoch: 5

Loss=0.5015882253646851 Batch_id=97  LR=0.00061 Accuracy=84.38: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.5374842882156372 Batch_id=19 LR=0.00061 Accuracy=82.18: 100%|██████████| 20/20 [00:04<00:00,  4.89it/s]

Adjusting learning rate of group 0 to 6.1565e-04.

Epoch: 6

Loss=0.4173966944217682 Batch_id=97  LR=0.00062 Accuracy=85.80: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.44997638463974 Batch_id=19 LR=0.00062 Accuracy=85.14: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.1802e-04.

Epoch: 7

Loss=0.4181428551673889 Batch_id=97  LR=0.00062 Accuracy=86.77: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.4139540195465088 Batch_id=19 LR=0.00062 Accuracy=85.77: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.2075e-04.

Epoch: 8

Loss=0.3991256356239319 Batch_id=97  LR=0.00062 Accuracy=87.66: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.36361372470855713 Batch_id=19 LR=0.00062 Accuracy=86.52: 100%|██████████| 20/20 [00:04<00:00,  4.87it/s]

Adjusting learning rate of group 0 to 6.2385e-04.

Epoch: 9

Loss=0.31517350673675537 Batch_id=97  LR=0.00062 Accuracy=89.02: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.4442049562931061 Batch_id=19 LR=0.00062 Accuracy=86.33: 100%|██████████| 20/20 [00:04<00:00,  4.95it/s]

Adjusting learning rate of group 0 to 6.2731e-04.

Epoch: 10

Loss=0.3664068281650543 Batch_id=97  LR=0.00063 Accuracy=89.24: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.30936217308044434 Batch_id=19 LR=0.00063 Accuracy=87.58: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.3113e-04.

Epoch: 11

Loss=0.2899329662322998 Batch_id=97  LR=0.00063 Accuracy=90.03: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.3378755748271942 Batch_id=19 LR=0.00063 Accuracy=87.57: 100%|██████████| 20/20 [00:04<00:00,  4.92it/s]

Adjusting learning rate of group 0 to 6.3532e-04.

Epoch: 12

Loss=0.3401070535182953 Batch_id=97  LR=0.00064 Accuracy=90.40: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.3330799639225006 Batch_id=19 LR=0.00064 Accuracy=87.63: 100%|██████████| 20/20 [00:04<00:00,  4.89it/s]

Adjusting learning rate of group 0 to 6.3987e-04.

Epoch: 13

Loss=0.2962125241756439 Batch_id=97  LR=0.00064 Accuracy=90.91: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.3002583682537079 Batch_id=19 LR=0.00064 Accuracy=88.86: 100%|██████████| 20/20 [00:04<00:00,  4.88it/s]

Adjusting learning rate of group 0 to 6.4479e-04.

Epoch: 14

Loss=0.2694096863269806 Batch_id=97  LR=0.00064 Accuracy=90.84: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.39415431022644043 Batch_id=19 LR=0.00064 Accuracy=88.98: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.5007e-04.

Epoch: 15

Loss=0.20260079205036163 Batch_id=97  LR=0.00065 Accuracy=91.46: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.32703331112861633 Batch_id=19 LR=0.00065 Accuracy=89.28: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.5571e-04.

Epoch: 16

Loss=0.29530856013298035 Batch_id=97  LR=0.00066 Accuracy=91.72: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.415722131729126 Batch_id=19 LR=0.00066 Accuracy=87.61: 100%|██████████| 20/20 [00:04<00:00,  4.90it/s]

Adjusting learning rate of group 0 to 6.6171e-04.

Epoch: 17

Loss=0.2937343120574951 Batch_id=97  LR=0.00066 Accuracy=91.93: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.33124756813049316 Batch_id=19 LR=0.00066 Accuracy=88.53: 100%|██████████| 20/20 [00:04<00:00,  4.85it/s]

Adjusting learning rate of group 0 to 6.6808e-04.

Epoch: 18

Loss=0.2692720293998718 Batch_id=97  LR=0.00067 Accuracy=92.08: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.27197396755218506 Batch_id=19 LR=0.00067 Accuracy=89.11: 100%|██████████| 20/20 [00:04<00:00,  4.88it/s]

Adjusting learning rate of group 0 to 6.7481e-04.

Epoch: 19

Loss=0.21367216110229492 Batch_id=97  LR=0.00067 Accuracy=92.61: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.37664076685905457 Batch_id=19 LR=0.00067 Accuracy=89.27: 100%|██████████| 20/20 [00:04<00:00,  4.91it/s]

Adjusting learning rate of group 0 to 6.8189e-04.

Epoch: 20

Loss=0.2836003303527832 Batch_id=97  LR=0.00068 Accuracy=92.63: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.3135916590690613 Batch_id=19 LR=0.00068 Accuracy=88.85: 100%|██████████| 20/20 [00:04<00:00,  4.84it/s]

Adjusting learning rate of group 0 to 6.8935e-04.

Epoch: 21

Loss=0.20260970294475555 Batch_id=97  LR=0.00069 Accuracy=92.60: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.31954193115234375 Batch_id=19 LR=0.00069 Accuracy=88.84: 100%|██████████| 20/20 [00:04<00:00,  4.89it/s]

Adjusting learning rate of group 0 to 6.9716e-04.

Epoch: 22

Loss=0.24486757814884186 Batch_id=97  LR=0.00070 Accuracy=92.89: 100%|██████████| 98/98 [00:53<00:00,  1.84it/s]
Loss=0.2755872309207916 Batch_id=19 LR=0.00070 Accuracy=90.28: 100%|██████████| 20/20 [00:04<00:00,  4.89it/s]

Saving..
Adjusting learning rate of group 0 to 7.0533e-04.

Epoch: 23

Loss=0.2430505007505417 Batch_id=97  LR=0.00071 Accuracy=93.07: 100%|██████████| 98/98 [00:53<00:00,  1.83it/s]
Loss=0.28398263454437256 Batch_id=19 LR=0.00071 Accuracy=90.05: 100%|██████████| 20/20 [00:04<00:00,  4.89it/s]

Adjusting learning rate of group 0 to 7.1387e-04.

```

## Image Augmentation

```
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
```
