# Business Problem

1. Your new target is:
   1. 99.4% **(this must be consistently shown in your last few epochs, and not a one-time achievement)**
   2. Less than or equal to 15 Epochs
   3. Less than 10000 Parameters (additional points for doing this in less than 8000 pts)
2. Do this in exactly 3 steps
3. Each File must have "target, result, analysis" TEXT block (either at the start or the end)
4. You must convince why have you decided that your target should be what you have decided it to be, and your analysis MUST be correct. 
5. Evaluation is highly subjective, and if you target anything out of the order, marks will be deducted. 
6. Explain your 3 steps using these **target**, **results**, and **analysis** with **links** to your GitHub files (Colab files moved to GitHub). 
7. Keep Receptive field calculations handy for each of your models. 
8. If your GitHub folder structure or file_names are messy, -100. 
9. When ready, attempt S5-Assignment Solution

# Final Model

This is the stage-7 one and notbook link is https://github.com/amitkml/Transformer-DeepLearning/blob/main/Class-5-Coding-Drilldown/EVA7_Session5_Assignment_Stage_7_0.ipynb

**Target**:

- Achive 99.4% Test accuracy consistently from earlier Test Accuracy of 99.40 once

**Results**:

- Parameters: 7,712
- Best Training Accuracy: 99.37%
- Best Test Accuracy: 99.4000%

**Analysis**:

- Added image augmentation of rotation +4 to -4 degree instead of -3 to +3 from earlier mode.
- Reduced dropout from 0.1 to 0.08. This is being done to ensure my model gets more weights to predict.
- Have been able to reach 99.4% test accuracy consistently within 8K parameters

## Model Architecture and Data Augmentation

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
       BatchNorm2d-2            [-1, 8, 26, 26]              16
              ReLU-3            [-1, 8, 26, 26]               0
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]             720
       BatchNorm2d-6           [-1, 10, 24, 24]              20
              ReLU-7           [-1, 10, 24, 24]               0
            Conv2d-8           [-1, 20, 22, 22]           1,800
       BatchNorm2d-9           [-1, 20, 22, 22]              40
             ReLU-10           [-1, 20, 22, 22]               0
          Dropout-11           [-1, 20, 22, 22]               0
        MaxPool2d-12           [-1, 20, 11, 11]               0
           Conv2d-13           [-1, 10, 11, 11]             200
      BatchNorm2d-14           [-1, 10, 11, 11]              20
             ReLU-15           [-1, 10, 11, 11]               0
           Conv2d-16             [-1, 10, 9, 9]             900
      BatchNorm2d-17             [-1, 10, 9, 9]              20
             ReLU-18             [-1, 10, 9, 9]               0
          Dropout-19             [-1, 10, 9, 9]               0
           Conv2d-20             [-1, 20, 7, 7]           1,800
      BatchNorm2d-21             [-1, 20, 7, 7]              40
             ReLU-22             [-1, 20, 7, 7]               0
           Conv2d-23             [-1, 12, 7, 7]             240
      BatchNorm2d-24             [-1, 12, 7, 7]              24
             ReLU-25             [-1, 12, 7, 7]               0
        MaxPool2d-26             [-1, 12, 3, 3]               0
           Conv2d-27             [-1, 15, 1, 1]           1,620
      BatchNorm2d-28             [-1, 15, 1, 1]              30
             ReLU-29             [-1, 15, 1, 1]               0
        AvgPool2d-30             [-1, 15, 1, 1]               0
           Conv2d-31             [-1, 10, 1, 1]             150
================================================================
Total params: 7,712
Trainable params: 7,712
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.70
Params size (MB): 0.03
Estimated Total Size (MB): 0.73
----------------------------------------------------------------
```

## Model Performance

```
loss=0.0752125009894371 batch_id=468 Accuracy=92.02: 100%|██████████| 469/469 [00:29<00:00, 16.09it/s]

Test set: Average loss: 0.0592, Accuracy: 9812/10000 (98.1200%)

loss=0.01667499728500843 batch_id=468 Accuracy=97.97: 100%|██████████| 469/469 [00:29<00:00, 15.90it/s]

Test set: Average loss: 0.0472, Accuracy: 9845/10000 (98.4500%)

loss=0.042152900248765945 batch_id=468 Accuracy=98.42: 100%|██████████| 469/469 [00:29<00:00, 16.04it/s]

Test set: Average loss: 0.0353, Accuracy: 9887/10000 (98.8700%)

loss=0.03698549419641495 batch_id=468 Accuracy=98.68: 100%|██████████| 469/469 [00:29<00:00, 15.98it/s]

Test set: Average loss: 0.0344, Accuracy: 9885/10000 (98.8500%)

loss=0.025376802310347557 batch_id=468 Accuracy=98.77: 100%|██████████| 469/469 [00:29<00:00, 15.87it/s]

Test set: Average loss: 0.0309, Accuracy: 9905/10000 (99.0500%)

loss=0.07272309809923172 batch_id=468 Accuracy=98.89: 100%|██████████| 469/469 [00:29<00:00, 16.02it/s]

Test set: Average loss: 0.0281, Accuracy: 9903/10000 (99.0300%)

loss=0.015030262060463428 batch_id=468 Accuracy=98.95: 100%|██████████| 469/469 [00:29<00:00, 15.94it/s]

Test set: Average loss: 0.0289, Accuracy: 9910/10000 (99.1000%)

loss=0.005469195079058409 batch_id=468 Accuracy=98.98: 100%|██████████| 469/469 [00:29<00:00, 16.11it/s]

Test set: Average loss: 0.0278, Accuracy: 9912/10000 (99.1200%)

loss=0.012685845606029034 batch_id=468 Accuracy=99.01: 100%|██████████| 469/469 [00:29<00:00, 16.01it/s]

Test set: Average loss: 0.0265, Accuracy: 9921/10000 (99.2100%)

loss=0.0014446512795984745 batch_id=468 Accuracy=99.16: 100%|██████████| 469/469 [00:28<00:00, 16.18it/s]

Test set: Average loss: 0.0216, Accuracy: 9937/10000 (99.3700%)

loss=0.022858763113617897 batch_id=468 Accuracy=99.35: 100%|██████████| 469/469 [00:29<00:00, 16.07it/s]

Test set: Average loss: 0.0201, Accuracy: 9942/10000 (99.4200%)

loss=0.002393869450315833 batch_id=468 Accuracy=99.37: 100%|██████████| 469/469 [00:29<00:00, 15.94it/s]

Test set: Average loss: 0.0198, Accuracy: 9937/10000 (99.3700%)

loss=0.022298207506537437 batch_id=468 Accuracy=99.39: 100%|██████████| 469/469 [00:29<00:00, 15.92it/s]

Test set: Average loss: 0.0198, Accuracy: 9941/10000 (99.4100%)

loss=0.021335842087864876 batch_id=468 Accuracy=99.38: 100%|██████████| 469/469 [00:29<00:00, 15.90it/s]

Test set: Average loss: 0.0193, Accuracy: 9940/10000 (99.4000%)
```

# Network Design

## Stage-1

### Target:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop with decent accuracy

### Results:

- Parameters: 16,634
- Best Training Accuracy: 98.00
- Best Test Accuracy: 98.00

### Analysis:

- Model is working good having Test and Train accuracy running almost same. This means model does not have any sign of overfitting
- No of parameter is high compare to benchmarking of 10k
- Model Accuracy is not as per target set of 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)

## Stage-2

### Target
- Have Model Training and Test Accuracy to 99%
- Reduce number of model parameter to 9-10K

### Results
- Parameters: 10,970
- Best Training Accuracy: 99
- Best Test Accuracy: 99

### Analysis
- Model parameter has been reduced from 16940 to 10,970
- No of parameter is high compare to benchmarking of 10k
- Adding BN and Droput(in every conv block) has helped to improve the model
- Model Accuracy is not as per target set of 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)

## Stage-3

### Target
- Have Model Training and Test Accuracy abive 99%
- Reduce number of model parameter to 9-10K

### Results
- Parameters: 6,170
- Best Training Accuracy: 98.58
- Best Test Accuracy: 99

### Analysis
- Replaced 7x7 by following
    - AvgPool2d
    - Conv2d(1x1)
- Model did not reach benchmarking accuracy of 99.4
- Model learning seems to be OK and I dont see any need for higher no of parameters

## Stage 4

**Target**:

- Have Model Training and Test Accuracy abive 99%
- Add CLR
- Add image Augmentation

**Results**:

- Parameters: 7,914
- Best Training Accuracy:  99
- Best Test Accuracy: 99

**Analysis**:

- Have added one more convolution layer before GAP to increase the parameter as i am hoping that will allow me to cross 99.4
- Have used cyclic learning rate
- Have tried trng augmentation and did not help

## Stage 5

 **Target**:

- Have Model Training and Test Accuracy abive 99%
- Add CLR
- Add image Augmentation

**Results**:

- Parameters: 7,712
- Best Training Accuracy: 99.35
- Best Test Accuracy: 99.30

**Analysis**:

- Added 2nd Max Pooling
- Reduced AveragePooling kernel size 5 to 1
- Added CONV2D 1x1 kernel after average Pooling
- Added stepLR and found that step 4500 is giving better result. The  step 4500 is almost equal to 10 epoch and this means reduction of LR  after 10 epoch seems to be giving better result and have increased Test  accuracy from 99% to 99.30

## Stage 6

 **Target**:

- Achive 99.4% Test accuracy from earlier Test Accuracy of 99.30 based on stage 5 model

**Results**:

- Parameters: 7,712
- Best Training Accuracy: 99.37%
- Best Test Accuracy: 99.4000%

**Analysis**:

- Added image augementation of rotation +3 to -3 degree
- Have been able to reach 99.4% test accuracy within 8K parameters but  i want to achive multiple times this same accuracy to ensure this  accuracy is stable enough on test data.

## Stage 7

**Target**:

- Achive 99.4% Test accuracy consistently from earlier Test Accuracy of 99.40 once

**Results**:

- Parameters: 7,712
- Best Training Accuracy: 99.39%
- Best Test Accuracy: 99.4100%

**Analysis**:

- Added image augementation of rotation +4 to -4 degree
- Reduced dropout from 0.1 to 0.08. This is being done to ensure my model gets more weights to predict.
- Have been able to reach 99.4% test accuracy consistently within 8K parameters

