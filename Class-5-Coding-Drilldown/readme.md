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
