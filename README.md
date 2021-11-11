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
