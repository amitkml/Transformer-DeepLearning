
# Transformer Based Deep Learning Projects
This repo contains all the assignments from the course called EVA conducted by the 'The School Of AI'. Object detection, Object reconisation, segmentation, monocular depth estimation.

![im](https://aws1.discourse-cdn.com/business7/uploads/wandb/optimized/1X/592ff23fa447cd2f78c947a05b7f5a01e1944573_2_686x500.png)
- Basics of Python 
- Setting Up Basic Skeleton
- Pytorch Basics
- To achive 99.4% validation accuracy for MNIST DatSet
- The whole process- Coding drill
- Regularisation techinques on MNIST
- Advanced convolutions(depthwise seperable and dialated convolutions) on CIFAR 10 with advanced image augmentation like utout, coarseDropout
- Vision Transformer
- Training on TinyImageNet
- Object detection in YOLO
- Custom Object detection training in Yolo

![im](https://1.bp.blogspot.com/-_mnVfmzvJWc/X8gMzhZ7SkI/AAAAAAAAG24/8gW2AHEoqUQrBwOqjhYB37A7OOjNyKuNgCLcBGAsYHQ/s1600/image1.gif)
# [Deep Learning Do It Yourself!](https://dataflowr.github.io/website/#deep_learning_do_it_yourself)

# PyTorch vs TensorFlow in 2022
- [PyTorch vs TensorFlow in 2022](https://www.assemblyai.com/blog/pytorch-vs-tensorflow-in-2022/)
- [Deep Learning Latest Blog](https://jarvislabs.ai/blog)

# Fullstack Deep Learning
- [Fullstack Deep Learning](https://fullstackdeeplearning.com/)
- [Taking FastAI to Production](https://community.wandb.ai/t/taking-fastai-to-production/1705)

# Explaiable AI
- [Explianble AI by Lime](https://github.com/marcotcr/lime)

# Labelling

# Transformer-DeepLearning

EVA Transformerbased Deep Learning course. This repo contains all my nlp work and learning. Made public so that others can learn and get benefits.The repo will contain all my project related to NLP learning and vision model deployment using mediapipe.

## Pytorch Package Hierarchy
![Pytorch](https://manalelaidouni.github.io/assets/img/pexels/Pytorch-package-hierarchy.jpg)

## Network Visualization
- [Tools-to-Design-or-Visualize-Architecture-of-Neural-Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)

## Different types of Convolution
The Convolution operation reduces the spatial dimensions as we go deeper down the network and creates an abstract representation of the input image. This feature of CNN’s is very useful for tasks like image classification where you just have to predict whether a particular object is present in the input image or not.

### Dilated convolution
- dilated convolutions are used to increase the receptive field of the higher layers, compensating for the reduction in receptive field induced by removing subsampling.

![im](https://www.researchgate.net/publication/336002670/figure/fig1/AS:806667134455815@1569335840531/An-illustration-of-the-receptive-field-for-one-dilated-convolution-with-different.png)
![im](https://miro.medium.com/max/875/1*btockft7dtKyzwXqfq70_w.gif)

A scenario of dilated convolution for kernel size 3×3. From the top: (a) it is the situation of the standard convolutional layer when the dilation rate is (1,1). (b) when the dilation rate become (2,2) the receptive field increases. (c) in the last case, the dilation rate is (3,3) and the receptive field enlarges even more than situation b.

![im](https://www.researchgate.net/profile/Mohammad-Hamed-Mozaffari/publication/335390357/figure/fig2/AS:795761700794376@1566735782667/A-scenario-of-dilated-convolution-for-kernel-size-33-From-the-top-a-it-is-the.jpg)

### Transpose Convolution
The convolution feature might cause problems for tasks like Object Localization, Segmentation where the spatial dimensions of the object in the original image are necessary to predict the output bounding box or segment the object. To fix this problem various techniques are used such as fully convolutional neural networks where we preserve the input dimensions using ‘same’ padding. Though this technique solves the problem to a great extent, it also increases the computation cost as now the convolution operation has to be applied to original input dimensions throughout the network.

![im](https://miro.medium.com/max/875/1*faRskFzI7GtvNCLNeCN8cg.png)

In the first step, the input image is padded with zeros, while in the second step the kernel is placed on the padded input and slid across generating the output pixels as dot products of the kernel and the overlapped input region. The kernel is slid across the padded input by taking jumps of size defined by the stride. The convolutional layer usually does a down-sampling i.e. the spatial dimensions of the output are less than that of the input.
The animations below explain the working of convolutional layers for different values of stride and padding.

![im](https://miro.medium.com/max/1250/1*YvlCSNzDEBGEWkZWNffPvw.gif)

### Depthwise Separable Convolution

Depthwise Separable Convolutions
A lot about such convolutions published in the (Xception paper) or (MobileNet paper). Consist of:

Depthwise convolution, i.e. a spatial convolution performed independently over each channel of an input.
Pointwise convolution, i.e. a 1x1 convolution, projecting the channels output by the depthwise convolution onto a new channel space.
Difference between Inception module and separable convolutions:

Separable convolutions perform first channel-wise spatial convolution and then perform 1x1 convolution, whereas Inception performs the 1x1 convolution first.
depthwise separable convolutions are usually implemented without non-linearities.
![im](https://ikhlestov.github.io/images/ML_notes/convolutions/05_1_deepwise_convolutions.png)

# Advanced Image Augmentation

## Coarse Dropout and Cutout Augmentation for GPU/TPU
Coarse Dropout and Cutout augmentation are techniques to prevent overfitting and encourage generalization. They randomly remove rectangles from training images. By removing portions of the images, we challenge our models to pay attention to the entire image because it never knows what part of the image will be present. (This is similar and different to dropout layer within a CNN).
- Cutout is the technique of removing 1 large rectangle of random size
- Coarse dropout is the technique of removing many small rectanges of similar size.

Example. Drop 2% of all pixels by converting them to black pixels, but do that on a lower-resolution version of the image that has 50% of the original size, leading to 2x2 squares being dropped:

```
import imgaug.augmenters as iaa
aug = iaa.CoarseDropout(0.02, size_percent=0.5)
```

![im](https://imgaug.readthedocs.io/en/latest/_images/coarsedropout.jpg)

# Advanced Training Concepts

## Cyclic LR
Cyclic learning rates (and cyclic momentum, which usually goes hand-in-hand) is a learning rate scheduling technique for (1) faster training of a network and (2) a finer understanding of the optimal learning rate. Cyclic learning rates have an effect on the model training process known somewhat fancifully as "superconvergence".

To apply cyclic learning rate and cyclic momentum to a run, begin by specifying a minimum and maximum learning rate and a minimum and maximum momentum. Over the course of a training run, the learning rate will be inversely scaled from its minimum to its maximum value and then back again, while the inverse will occur with the momentum. At the very end of training the learning rate will be reduced even further, an order of magnitude or two below the minimum learning rate, in order to squeeze out the last bit of convergence.
The maximum should be the value picked with a learning rate finder procedure, and the minimum value can be ten times lower.
## OneCycleLR
Cyclic learning rates (and cyclic momentum, which usually goes hand-in-hand) is a learning rate scheduling technique for (1) faster training of a network and (2) a finer understanding of the optimal learning rate. Cyclic learning rates have an effect on the model training process known somewhat fancifully as "superconvergence".

To apply cyclic learning rate and cyclic momentum to a run, begin by specifying a minimum and maximum learning rate and a minimum and maximum momentum. Over the course of a training run, the learning rate will be inversely scaled from its minimum to its maximum value and then back again, while the inverse will occur with the momentum. At the very end of training the learning rate will be reduced even further, an order of magnitude or two below the minimum learning rate, in order to squeeze out the last bit of convergence.
The paper suggests the highest batch size value that can be fit into memory to be used as batch size. The author suggests , its reasonable to make combined run with CLR and Cyclic momentum with different values of weight decay to determine learning rate, momentum range and weigh decay simultaneously. The paper suggests to use values like 1e-3, 1e-4, 1e-5 and 0 to start with, if there is no notion of what is correct weight decay value. On the other hand, if we know , say 1e-4 is correct value, paper suggests to try 3 values bisecting the exponent( 3e-4, 1e-4 and 3e-5).

![im](https://miro.medium.com/max/510/1*VaHVbnxikt6KD5-etumSSw.png)

- [Pytprch LR Finder](https://github.com/davidtvs/pytorch-lr-finder)
- [OneCycleLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html)

## Bi-Linear Fine Grained Image classification
Fine-grain visual classification (FGVC) refers to the task of distinguishing the categories of the same class. ... Fine-grained visual classification of species or objects of any category is a herculean task for human beings and usually requires extensive domain knowledge to identify the species or objects correctly. Refer the video https://www.youtube.com/watch?v=s437TvBuziM

[Fine-Grained Image Classification (FGIC) with B-CNNs](https://wandb.ai/authors/bcnn/reports/Fine-Grained-Image-Classification-FGIC-with-B-CNNs---Vmlldzo0NDQ1Nzc)
![im](https://api.wandb.ai/files/authors/images/projects/209051/627087f0.png)

![im](http://vis-www.cs.umass.edu/bcnn/docs/teaser-bcnn.png)

![im](https://www.researchgate.net/profile/Zhiwu-Lu-2/publication/318204948/figure/fig1/AS:512607066628096@1499226456148/Fine-grained-classification-vs-general-image-classification-Finegrained-classification.png)
![im](https://www.inf-cv.uni-jena.de/dbvmedia/de/Research/Fine_grained+Recognition/Goering13_FGC_1Col-0-.png)

## Fine-Grained Image Classification for Crop Disease Based on Attention Mechanism
![im](https://www.frontiersin.org/files/Articles/600854/fpls-11-600854-HTML/image_m/fpls-11-600854-g004.jpg)

[Fine-Grained Image Classification for Crop Disease Based on Attention Mechanism](https://www.frontiersin.org/articles/10.3389/fpls.2020.600854/full)

## Receptive Field
he Receptive Field (RF) is defined as the size of the region in the input that produces the feature[3]. Basically, it is a measure of association of an output feature (of any layer) to the input region (patch). The idea of receptive fields applies to local operations (i.e. convolution, pooling).A convolutional unit only depends on a local region (patch) of the input. That’s why we never refer to the RF on fully connected layers since each unit has access to all the input region. To this end, our aim is to provide you an insight into this concept, in order to understand and analyze how deep convolutional networks work with local operations work. **In essence, there are a plethora of ways and tricks to increase the RF, that can be summarized as follows**:
- Add more convolutional layers (make the network deeper)
- Add pooling layers or higher stride convolutions (sub-sampling)
- Use dilated convolutions
- Depth-wise convolutions
Refer the article https://theaisummer.com/receptive-field/

[Understanding the receptive field of deep convolutional networks](https://theaisummer.com/receptive-field/)

![im](https://theaisummer.com/static/490be17ee7f19b78003c3fdf5a6bbafc/83b75/receptive-field-in-convolutional-networks.png)
