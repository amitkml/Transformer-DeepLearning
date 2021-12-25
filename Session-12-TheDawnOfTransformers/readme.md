# Spatial Transformer

## Assignment Details

This Assignment talks about the Spatial Transformer Network. This concept is based on Attention Mechanism. This network is based on the CIFAR10 database which allow a neural network to learn how to perform spatial transformations on the input image in order to enhance the geometric invariance of the model.

**It has the following steps**

1. It loads the CIFAR10 data and apply some augmentation to it
2. Then it defines a network for 
   1. Special Transformation Localization 
   2. Regressor for the Affine Matrix 
   3. Spatial Transformer Forward Network 
   4. Transform Input forward function
3. Then we train the model using the data for 50 epochs
4. Then we test the model to find out the accuracy of Spatial Transofrmer
5. Then we visualize to see the original input and translated output

**Implemented [EVA7_SpacialTransformers_CIFAR10](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-12-TheDawnOfTransformers/SpacialTransformers/EVA7_SpacialTransformers_CIFAR10.ipynb) for CIFAR10 trained for 50 Epochs. Readme can be found from [Spatial Transformer Analysis](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-12-TheDawnOfTransformers/SpacialTransformers/readme.md)**


## What is Spatial Transformer?

A  Spatial Transformer Network (STN) is a learnable module that can be  placed in a Convolutional Neural Network (CNN), to increase the spatial  invariance in an efficient manner. Spatial invariance refers to the  invariance of the model towards spatial transformations of images such  as rotation, translation and scaling. Invariance is the ability of the  model to recognize and identify features even when the input is  transformed or slightly modified. Spatial Transformers can be placed  into CNNs to benefit various tasks. One example is image classification.  

Suppose the task is to perform classification of handwritten digits,  where the position, size and orientation of the digit in each sample  varies significantly. A spatial transformer crops out, transforms and  scales the region of interest in the sample. Now a CNN can perform the  task of classification.

![img](https://miro.medium.com/max/1400/1*Mq98rPf11Dk7vxL7OuDZ7A.png)

A Spatial Transformer Network consists of 3 main components:

- **Localization Network** : This network takes a 4D tensor representation of a batch of images (Width x Height x Channels x Batch_Size according to Flux conventions) as input. It is a simple neural network with a few convolution layers and a few dense layers. It predicts the parameters of transformation as output.
- **Sampling Grid Generator :** The transformation parameters predicted by the localization net are used in the form of an affine transformation matrix of size 2 x 3 for each image in the batch. An affine transformation is one which preserves points, straight lines and planes. Parallel lines remain parallel after affine  transformation. Rotation, scaling and translation are all affine transformations.
- **Bilinear Interpolation on transformed indices :** Now the indices and axes of the image have undergone an affine transformation. So its pixels have moved around. For example a point (1,
  1) after rotation of axes by 45 degrees counter clockwise becomes (√2, 0). So to find the pixel value at the transformed point we need to perform bilinear interpolation using the four nearest pixel values.



# Vision Transformer

## Transformers in Computer vision

Now that we know transformers are very interesting, there is still a problem in computer vision applications. Indeed, just like the popular saying “a picture is worth a thousand words,” pictures contain much more information than sentences, so we have to adapt the basic transformer’s architecture to process images effectively. This is what this paper is all about.

![img](https://cdn-images-1.medium.com/max/720/1*JS21YKMUuZ6i24Y9ozpNQQ.gif)Vision transformers’ complexity. Image by [Davide Coccomini](https://towardsdatascience.com/transformers-an-exciting-revolution-from-text-to-videos-dc70a15e617b) reposted with permission.

This is due to the fact that the computational complexity of its self-attention is quadratic to image size. Thus exploding the computation time and memory needs. Instead, the researchers replaced this quadratic computational complexity with a linear computational complexity to image size.

![img](https://cdn-images-1.medium.com/max/720/1*nzaAUXFzsKIr2t0u3SOPHQ.gif)

## What is Vision Transformer?

As a first step in this direction, we present the [Vision Transformer](https://arxiv.org/abs/2010.11929) (ViT), a vision model based as closely as possible on the [Transformer](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) architecture. ViT represents an input image as a sequence of image patches, similar to the sequence of word embeddings used when applying Transformers to text, and directly predicts class labels for the image. High level steps are:

**Detailed analysis on VIT been documented into [VIT analysis](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-12-TheDawnOfTransformers/VisionTransformers/readme.md).**

The overall architecture of the vision transformer model is given as follows in a step-by-step manner:

1. Split an image into patches (fixed sizes)
2. Flatten the image patches
3. Create lower-dimensional linear embeddings from these flattened image patches
4. Include positional embeddings
5. Feed the sequence as an input to a state-of-the-art transformer encoder
6. Pre-train the ViT model with image labels, which is then fully supervised on a big dataset
7. Fine-tune on the downstream dataset for image classification

A few things to remember are:

- In ViT, we represent an image as a sequence of patches .
- The architecture resembles the original Transformer from the famous “Attention is all you need” paper.
- The model is trained using a labeled dataset following a fully-supervised paradigm.
- It is usually fine-tuned on the downstream dataset for image classification.

![im](https://1.bp.blogspot.com/-_mnVfmzvJWc/X8gMzhZ7SkI/AAAAAAAAG24/8gW2AHEoqUQrBwOqjhYB37A7OOjNyKuNgCLcBGAsYHQ/s1600/image1.gif)
