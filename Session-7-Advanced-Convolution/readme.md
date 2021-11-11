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

There are two main parts to a CNN architecture

1. A convolution tool that separates and identifies the various features of the image for analysis in a process called as Feature Extraction.
2. A fully connected layer that utilizes the output from the convolution process and predicts the class of the image based on the features extracted in previous stages.

![img](https://miro.medium.com/max/1162/1*l9c8f0FRlJQ06IvPNLeXJw.png)

CNN Architecture

There are mainly three types of layers:

1. Convolutional Layer : Used to extract features from the image. It creates an MxM matrix filter that slides over the image and uses dot product with the input of the image. This gives us a Feature Map that gives us information about the image such as corners, edges. This is then fed to the other layers to learn more about the image.
2. Pooling Layer : The main goal of this layer is to reduce the convoluted size of the feature map and to reduce Computational costs. This is done by decreasing the connections between layers. It acts as a bridge between Convolutional Layer and the FC layer.
3. Fully Connected Layer : This layer consists of weights and biases along with neurons to connect the various layers. The input image is flattened and fed to this layer. Mathematical operations are then used to do classification of images.
4. Dropout Layer : Usually when all features are connected to FC Layer, it can cause over fitting wherein model performs well on training set but not on the test set. On passing a dropout of 0.3, 30% of neurons are dropped randomly from the network
5. Activation Functions : These are used to learn and approximate any kind of continuous and complex relations between variables of the network. It decides when the variables should fire and when they shouldn't. It also adds non-linearity to the model.