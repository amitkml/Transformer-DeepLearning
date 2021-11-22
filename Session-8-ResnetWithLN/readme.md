# Assignment Details



EARLY SUBMISSIONS:

1. Train for **20** Epochs
2. **10** misclassified images
3. **10** GradCam output on **ANY misclassified images**
4. Apply these transforms while training:
   1. RandomCrop(32, padding=4)
   2. CutOut(16x16)

LATE SUBMISSIONS:

1. Train for 40 Epochs

2. **20** misclassified images

3. **20** GradCam output on the **SAME misclassified images**

4. Apply these transforms while training:

   1. RandomCrop(32, padding=4)
   2. CutOut(16x16)
   3. **Rotate(±5°)**

5. **Must** use ReduceLROnPlateau

6. **Must** use LayerNormalization ONLY

   

## Model Performance

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-8-ResnetWithLN/model-performance.JPG?raw=true)

## Model sample misclassification

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-8-ResnetWithLN/misclassification-sample.JPG?raw=true)



## Gradcam from Model

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-8-ResnetWithLN/gradcam-image.JPG?raw=true)

## Model Training Log

```
Epoch: 0

Loss=2.0108983516693115 Batch_id=390  LR=0.10000 Accuracy=18.70: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=1.8305598497390747 Batch_id=99 LR=0.10000 Accuracy=32.05: 100%|██████████| 100/100 [00:09<00:00, 10.75it/s]

Saving..

Epoch: 1

Loss=1.8560717105865479 Batch_id=390  LR=0.10000 Accuracy=28.73: 100%|██████████| 391/391 [02:28<00:00,  2.64it/s]
Loss=1.607298731803894 Batch_id=99 LR=0.10000 Accuracy=41.60: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Saving..

Epoch: 2

Loss=1.8600833415985107 Batch_id=390  LR=0.10000 Accuracy=32.82: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=1.5231950283050537 Batch_id=99 LR=0.10000 Accuracy=46.34: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Saving..

Epoch: 3

Loss=1.5457738637924194 Batch_id=390  LR=0.10000 Accuracy=38.83: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=1.6723936796188354 Batch_id=99 LR=0.10000 Accuracy=51.73: 100%|██████████| 100/100 [00:09<00:00, 10.70it/s]

Saving..

Epoch: 4

Loss=1.48893404006958 Batch_id=390  LR=0.10000 Accuracy=45.48: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=1.1611394882202148 Batch_id=99 LR=0.10000 Accuracy=61.07: 100%|██████████| 100/100 [00:09<00:00, 10.70it/s]

Saving..

Epoch: 5

Loss=1.456124186515808 Batch_id=390  LR=0.10000 Accuracy=50.94: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.7949082851409912 Batch_id=99 LR=0.10000 Accuracy=67.48: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Saving..

Epoch: 6

Loss=1.4177348613739014 Batch_id=390  LR=0.10000 Accuracy=54.71: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=1.1113102436065674 Batch_id=99 LR=0.10000 Accuracy=66.46: 100%|██████████| 100/100 [00:09<00:00, 10.77it/s]

Epoch     7: reducing learning rate of group 0 to 1.0000e-02.

Epoch: 7

Loss=0.9913004636764526 Batch_id=390  LR=0.01000 Accuracy=61.57: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.677293062210083 Batch_id=99 LR=0.01000 Accuracy=78.61: 100%|██████████| 100/100 [00:09<00:00, 10.69it/s]

Saving..

Epoch: 8

Loss=0.9674711227416992 Batch_id=390  LR=0.01000 Accuracy=63.21: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.6710057854652405 Batch_id=99 LR=0.01000 Accuracy=79.97: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Saving..

Epoch: 9

Loss=1.087589144706726 Batch_id=390  LR=0.01000 Accuracy=64.49: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.61716228723526 Batch_id=99 LR=0.01000 Accuracy=81.26: 100%|██████████| 100/100 [00:09<00:00, 10.70it/s]

Saving..

Epoch: 10

Loss=1.0667568445205688 Batch_id=390  LR=0.01000 Accuracy=65.03: 100%|██████████| 391/391 [02:27<00:00,  2.64it/s]
Loss=0.6301790475845337 Batch_id=99 LR=0.01000 Accuracy=81.21: 100%|██████████| 100/100 [00:09<00:00, 10.79it/s]

Epoch    11: reducing learning rate of group 0 to 1.0000e-03.

Epoch: 11

Loss=1.0217616558074951 Batch_id=390  LR=0.00100 Accuracy=65.99: 100%|██████████| 391/391 [02:27<00:00,  2.65it/s]
Loss=0.6174797415733337 Batch_id=99 LR=0.00100 Accuracy=82.70: 100%|██████████| 100/100 [00:09<00:00, 10.81it/s]

Saving..

Epoch: 12

Loss=0.9976484179496765 Batch_id=390  LR=0.00100 Accuracy=66.80: 100%|██████████| 391/391 [02:27<00:00,  2.65it/s]
Loss=0.6092908382415771 Batch_id=99 LR=0.00100 Accuracy=82.92: 100%|██████████| 100/100 [00:09<00:00, 10.78it/s]

Saving..

Epoch: 13

Loss=0.9780610799789429 Batch_id=390  LR=0.00100 Accuracy=67.12: 100%|██████████| 391/391 [02:28<00:00,  2.64it/s]
Loss=0.6009781360626221 Batch_id=99 LR=0.00100 Accuracy=83.20: 100%|██████████| 100/100 [00:09<00:00, 10.74it/s]

Saving..

Epoch: 14

Loss=1.1555191278457642 Batch_id=390  LR=0.00100 Accuracy=67.70: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5756196975708008 Batch_id=99 LR=0.00100 Accuracy=83.18: 100%|██████████| 100/100 [00:09<00:00, 10.68it/s]

Epoch    15: reducing learning rate of group 0 to 1.0000e-04.

Epoch: 15

Loss=0.7121686339378357 Batch_id=390  LR=0.00010 Accuracy=67.29: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5724261999130249 Batch_id=99 LR=0.00010 Accuracy=83.28: 100%|██████████| 100/100 [00:09<00:00, 10.74it/s]

Saving..

Epoch: 16

Loss=1.0307241678237915 Batch_id=390  LR=0.00010 Accuracy=67.39: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5764091610908508 Batch_id=99 LR=0.00010 Accuracy=83.21: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Epoch    17: reducing learning rate of group 0 to 1.0000e-05.

Epoch: 17

Loss=0.9017132520675659 Batch_id=390  LR=0.00001 Accuracy=67.78: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5774734616279602 Batch_id=99 LR=0.00001 Accuracy=83.16: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Epoch    18: reducing learning rate of group 0 to 1.0000e-06.

Epoch: 18

Loss=0.869956374168396 Batch_id=390  LR=0.00000 Accuracy=67.59: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.5775911211967468 Batch_id=99 LR=0.00000 Accuracy=83.27: 100%|██████████| 100/100 [00:09<00:00, 10.73it/s]

Epoch    19: reducing learning rate of group 0 to 1.0000e-07.

Epoch: 19

Loss=0.9102106094360352 Batch_id=390  LR=0.00000 Accuracy=67.68: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5821373462677002 Batch_id=99 LR=0.00000 Accuracy=83.10: 100%|██████████| 100/100 [00:09<00:00, 10.71it/s]

Epoch    20: reducing learning rate of group 0 to 1.0000e-08.

Epoch: 20

Loss=0.9675021171569824 Batch_id=390  LR=0.00000 Accuracy=67.52: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.5748768448829651 Batch_id=99 LR=0.00000 Accuracy=83.22: 100%|██████████| 100/100 [00:09<00:00, 10.72it/s]


Epoch: 21

Loss=0.9375483393669128 Batch_id=390  LR=0.00000 Accuracy=67.65: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5819188952445984 Batch_id=99 LR=0.00000 Accuracy=83.19: 100%|██████████| 100/100 [00:09<00:00, 10.74it/s]


Epoch: 22

Loss=1.056692123413086 Batch_id=390  LR=0.00000 Accuracy=67.62: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5788881778717041 Batch_id=99 LR=0.00000 Accuracy=83.23: 100%|██████████| 100/100 [00:09<00:00, 10.75it/s]


Epoch: 23

Loss=0.9058238863945007 Batch_id=390  LR=0.00000 Accuracy=67.86: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5771032571792603 Batch_id=99 LR=0.00000 Accuracy=83.13: 100%|██████████| 100/100 [00:09<00:00, 10.72it/s]


Epoch: 24

Loss=0.8146520853042603 Batch_id=390  LR=0.00000 Accuracy=67.44: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.571803092956543 Batch_id=99 LR=0.00000 Accuracy=83.25: 100%|██████████| 100/100 [00:09<00:00, 10.75it/s]


Epoch: 25

Loss=0.6905322074890137 Batch_id=390  LR=0.00000 Accuracy=67.51: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5776333212852478 Batch_id=99 LR=0.00000 Accuracy=83.21: 100%|██████████| 100/100 [00:09<00:00, 10.70it/s]


Epoch: 26

Loss=0.8831550478935242 Batch_id=390  LR=0.00000 Accuracy=67.38: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5779592990875244 Batch_id=99 LR=0.00000 Accuracy=83.21: 100%|██████████| 100/100 [00:09<00:00, 10.74it/s]


Epoch: 27

Loss=0.8018416166305542 Batch_id=390  LR=0.00000 Accuracy=67.40: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5739357471466064 Batch_id=99 LR=0.00000 Accuracy=83.16: 100%|██████████| 100/100 [00:09<00:00, 10.68it/s]


Epoch: 28

Loss=0.8162808418273926 Batch_id=390  LR=0.00000 Accuracy=67.84: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5848514437675476 Batch_id=99 LR=0.00000 Accuracy=83.13: 100%|██████████| 100/100 [00:09<00:00, 10.75it/s]


Epoch: 29

Loss=1.1711844205856323 Batch_id=390  LR=0.00000 Accuracy=67.53: 100%|██████████| 391/391 [02:29<00:00,  2.61it/s]
Loss=0.5824499726295471 Batch_id=99 LR=0.00000 Accuracy=83.11: 100%|██████████| 100/100 [00:09<00:00, 10.75it/s]


Epoch: 30

Loss=0.971813976764679 Batch_id=390  LR=0.00000 Accuracy=67.77: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.5777608752250671 Batch_id=99 LR=0.00000 Accuracy=83.21: 100%|██████████| 100/100 [00:09<00:00, 10.71it/s]


Epoch: 31

Loss=0.7871803045272827 Batch_id=390  LR=0.00000 Accuracy=67.25: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5818784236907959 Batch_id=99 LR=0.00000 Accuracy=83.13: 100%|██████████| 100/100 [00:09<00:00, 10.67it/s]


Epoch: 32

Loss=0.8218854069709778 Batch_id=390  LR=0.00000 Accuracy=67.41: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5731958150863647 Batch_id=99 LR=0.00000 Accuracy=83.21: 100%|██████████| 100/100 [00:09<00:00, 10.75it/s]


Epoch: 33

Loss=1.1211961507797241 Batch_id=390  LR=0.00000 Accuracy=67.59: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5840232372283936 Batch_id=99 LR=0.00000 Accuracy=83.15: 100%|██████████| 100/100 [00:09<00:00, 10.72it/s]


Epoch: 34

Loss=0.734233558177948 Batch_id=390  LR=0.00000 Accuracy=67.65: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5795077681541443 Batch_id=99 LR=0.00000 Accuracy=83.12: 100%|██████████| 100/100 [00:09<00:00, 10.78it/s]


Epoch: 35

Loss=0.9029927253723145 Batch_id=390  LR=0.00000 Accuracy=67.70: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.5802488923072815 Batch_id=99 LR=0.00000 Accuracy=83.19: 100%|██████████| 100/100 [00:09<00:00, 10.63it/s]


Epoch: 36

Loss=0.7969058752059937 Batch_id=390  LR=0.00000 Accuracy=67.36: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.5762954354286194 Batch_id=99 LR=0.00000 Accuracy=83.16: 100%|██████████| 100/100 [00:09<00:00, 10.66it/s]


Epoch: 37

Loss=0.6979414224624634 Batch_id=390  LR=0.00000 Accuracy=67.42: 100%|██████████| 391/391 [02:28<00:00,  2.62it/s]
Loss=0.5809479355812073 Batch_id=99 LR=0.00000 Accuracy=83.16: 100%|██████████| 100/100 [00:09<00:00, 10.66it/s]


Epoch: 38

Loss=1.0157110691070557 Batch_id=390  LR=0.00000 Accuracy=67.72: 100%|██████████| 391/391 [02:28<00:00,  2.63it/s]
Loss=0.5761234760284424 Batch_id=99 LR=0.00000 Accuracy=83.22: 100%|██████████| 100/100 [00:09<00:00, 10.65it/s]


Epoch: 39

Loss=1.0903851985931396 Batch_id=390  LR=0.00000 Accuracy=67.72: 100%|██████████| 391/391 [02:29<00:00,  2.62it/s]
Loss=0.5803743600845337 Batch_id=99 LR=0.00000 Accuracy=83.06: 100%|██████████| 100/100 [00:09<00:00, 10.65it/s]

============================================================ Training and Testing Performance ================================
```

## Resnet Model with Layer Normalization

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
         LayerNorm-2           [-1, 64, 32, 32]         131,072
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,304,906
Trainable params: 11,304,906
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 43.12
Estimated Total Size (MB): 54.39
----------------------------------------------------------------
```

