# Repo Purpose

This will be a generic repo which will host different pytorch model and associated utilities
- image transforms
- gradcam
- misclassification code,
- tensorboard related stuff
- advanced training policies

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Training

```
model_main.run_experiments_ln()

model_main.run_experiments()
```

## Accuracy

| Model                                                        | Accuracy |
| ------------------------------------------------------------ | -------- |
| [ResNet18](https://github.com/amitkml/Transformer-DeepLearning/blob/main/ConvModelAdvancedTraining/models/resnet.py) | 87.90    |
| [Resnet18 with LayerNorm](https://github.com/amitkml/Transformer-DeepLearning/blob/main/ConvModelAdvancedTraining/models/resnet_ln.py) | 83.25    |
|                                                              |          |