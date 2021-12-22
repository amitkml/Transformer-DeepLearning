# Spatial Transformer

The spatial transformer module consists of layers of neural networks that can spatially transform an image. These spatial transformations include cropping, scaling, rotations, and translations etc

CNNs perform poorly when the input data contains so much variation. One of the solutions to this is the max-pooling layer. But then again, max-pooling layers do no make the CNN invariant to large transformations in the input data.

This gives rise to the concept of Spatial Transformer Networks. In STNs, the transformer module knows where to apply the transformation to properly scale, resize, and crop and image. We can apply the STN module to the input data directly, or even to the feature maps (output of a convolution layer). In simple words, we can say that the spatial transformer module acts as an attention mechanism and knows where to focus on the input data.

## Architecture

The architecture of a Spatial Transformer Network is based on three important parts.

- The localization network.
- Parameterized sampling grid.
- Differentiable image sampling.

[![image](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-19_at_5.48.34_PM_vFLk7jR.png)](https://user-images.githubusercontent.com/42609155/127073287-08c80ce8-9686-4bdc-9933-cc6801f0f3cb.png)

### Localisation Network

The localization network takes the input feature map and outputs the parameters of the spatial transformations that should be applied to the feature map. The localization network is a very simple stacking of convolutional layers.

In the above figuare, U is the feature map input to the localization network. It outputs θ which are the transformation parameters that are regressed from the localization network. The final regression layers are fully-connected linear layers. Tθ is the transformation operation using the parameters θ.

### Parameterized Sampling Grid

Parameterized Sampling Grid mainly generates a sampling grid that is consistent with the picture pixels, and multiplies it with theta matrix to gradually learn to fully correspond to the tilt recognition object

### Differentiable image sampling.

Differentable Image Sampling is mainly used to obtain the original image pixels corresponding to the sampling points to form a V feature map to complete the output of the V feature map.

## Model Architecture

```
Net(
  (conv1): Conv2d(3, 16, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (conv2_drop): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=800, out_features=1024, bias=True)
  (fc2): Linear(in_features=1024, out_features=10, bias=True)
  (localization): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1))
    (1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (2): ReLU(inplace=True)
    (3): Conv2d(64, 128, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc_loc): Sequential(
    (0): Linear(in_features=2048, out_features=256, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=256, out_features=6, bias=True)
  )
)
```



## Training and Validation Log

```
Test set: Average loss: 1.8298, Accuracy: 3482/10000 (35%)

Train Epoch: 2 [0/50000 (0%)]	Loss: 1.801602
Train Epoch: 2 [32000/50000 (64%)]	Loss: 1.923332

Test set: Average loss: 1.6396, Accuracy: 4220/10000 (42%)

Train Epoch: 3 [0/50000 (0%)]	Loss: 1.732196
Train Epoch: 3 [32000/50000 (64%)]	Loss: 1.696665

Test set: Average loss: 1.5122, Accuracy: 4646/10000 (46%)

Train Epoch: 4 [0/50000 (0%)]	Loss: 1.794338
Train Epoch: 4 [32000/50000 (64%)]	Loss: 1.697090

Test set: Average loss: 1.4630, Accuracy: 4640/10000 (46%)

Train Epoch: 5 [0/50000 (0%)]	Loss: 1.515351
Train Epoch: 5 [32000/50000 (64%)]	Loss: 1.562161

Test set: Average loss: 1.3827, Accuracy: 5043/10000 (50%)

Train Epoch: 6 [0/50000 (0%)]	Loss: 1.446218
Train Epoch: 6 [32000/50000 (64%)]	Loss: 1.353426

Test set: Average loss: 1.4037, Accuracy: 5036/10000 (50%)

Train Epoch: 7 [0/50000 (0%)]	Loss: 1.566134
Train Epoch: 7 [32000/50000 (64%)]	Loss: 1.484562

Test set: Average loss: 1.3942, Accuracy: 5024/10000 (50%)

Train Epoch: 8 [0/50000 (0%)]	Loss: 1.557762
Train Epoch: 8 [32000/50000 (64%)]	Loss: 1.304269

Test set: Average loss: 1.3376, Accuracy: 5181/10000 (52%)

Train Epoch: 9 [0/50000 (0%)]	Loss: 1.277409
Train Epoch: 9 [32000/50000 (64%)]	Loss: 1.461986

Test set: Average loss: 1.4398, Accuracy: 5037/10000 (50%)

Train Epoch: 10 [0/50000 (0%)]	Loss: 1.892708
Train Epoch: 10 [32000/50000 (64%)]	Loss: 1.205936

Test set: Average loss: 1.2617, Accuracy: 5617/10000 (56%)

Train Epoch: 11 [0/50000 (0%)]	Loss: 1.465691
Train Epoch: 11 [32000/50000 (64%)]	Loss: 1.306489

Test set: Average loss: 1.2621, Accuracy: 5539/10000 (55%)

Train Epoch: 12 [0/50000 (0%)]	Loss: 1.158612
Train Epoch: 12 [32000/50000 (64%)]	Loss: 1.300986

Test set: Average loss: 1.1684, Accuracy: 5926/10000 (59%)

Train Epoch: 13 [0/50000 (0%)]	Loss: 1.126827
Train Epoch: 13 [32000/50000 (64%)]	Loss: 1.334673

Test set: Average loss: 1.1873, Accuracy: 5884/10000 (59%)

Train Epoch: 14 [0/50000 (0%)]	Loss: 1.481174
Train Epoch: 14 [32000/50000 (64%)]	Loss: 1.171596

Test set: Average loss: 1.1657, Accuracy: 5901/10000 (59%)

Train Epoch: 15 [0/50000 (0%)]	Loss: 1.193457
Train Epoch: 15 [32000/50000 (64%)]	Loss: 1.208172

Test set: Average loss: 1.1201, Accuracy: 6129/10000 (61%)

Train Epoch: 16 [0/50000 (0%)]	Loss: 1.118583
Train Epoch: 16 [32000/50000 (64%)]	Loss: 0.998188

Test set: Average loss: 1.1168, Accuracy: 6166/10000 (62%)

Train Epoch: 17 [0/50000 (0%)]	Loss: 1.259168
Train Epoch: 17 [32000/50000 (64%)]	Loss: 1.224004

Test set: Average loss: 1.0894, Accuracy: 6258/10000 (63%)

Train Epoch: 18 [0/50000 (0%)]	Loss: 1.245496
Train Epoch: 18 [32000/50000 (64%)]	Loss: 1.076321

Test set: Average loss: 1.0739, Accuracy: 6287/10000 (63%)

Train Epoch: 19 [0/50000 (0%)]	Loss: 1.082244
Train Epoch: 19 [32000/50000 (64%)]	Loss: 1.373497

Test set: Average loss: 1.2351, Accuracy: 5806/10000 (58%)

Train Epoch: 20 [0/50000 (0%)]	Loss: 1.067806
Train Epoch: 20 [32000/50000 (64%)]	Loss: 0.920418

Test set: Average loss: 1.0903, Accuracy: 6286/10000 (63%)

Train Epoch: 21 [0/50000 (0%)]	Loss: 0.925776
Train Epoch: 21 [32000/50000 (64%)]	Loss: 1.339825

Test set: Average loss: 1.2194, Accuracy: 5744/10000 (57%)

Train Epoch: 22 [0/50000 (0%)]	Loss: 1.330925
Train Epoch: 22 [32000/50000 (64%)]	Loss: 0.987479

Test set: Average loss: 1.0655, Accuracy: 6373/10000 (64%)

Train Epoch: 23 [0/50000 (0%)]	Loss: 1.011175
Train Epoch: 23 [32000/50000 (64%)]	Loss: 1.013353

Test set: Average loss: 1.0480, Accuracy: 6377/10000 (64%)

Train Epoch: 24 [0/50000 (0%)]	Loss: 1.236474
Train Epoch: 24 [32000/50000 (64%)]	Loss: 0.973540

Test set: Average loss: 1.0552, Accuracy: 6407/10000 (64%)

Train Epoch: 25 [0/50000 (0%)]	Loss: 1.064097
Train Epoch: 25 [32000/50000 (64%)]	Loss: 1.087154

Test set: Average loss: 1.0347, Accuracy: 6433/10000 (64%)

Train Epoch: 26 [0/50000 (0%)]	Loss: 0.961598
Train Epoch: 26 [32000/50000 (64%)]	Loss: 1.011200

Test set: Average loss: 1.1642, Accuracy: 5968/10000 (60%)

Train Epoch: 27 [0/50000 (0%)]	Loss: 1.258165
Train Epoch: 27 [32000/50000 (64%)]	Loss: 0.973740

Test set: Average loss: 1.0876, Accuracy: 6225/10000 (62%)

Train Epoch: 28 [0/50000 (0%)]	Loss: 0.746194
Train Epoch: 28 [32000/50000 (64%)]	Loss: 0.916931

Test set: Average loss: 1.0475, Accuracy: 6439/10000 (64%)

Train Epoch: 29 [0/50000 (0%)]	Loss: 0.811375
Train Epoch: 29 [32000/50000 (64%)]	Loss: 0.964783

Test set: Average loss: 1.0427, Accuracy: 6437/10000 (64%)

Train Epoch: 30 [0/50000 (0%)]	Loss: 0.803833
Train Epoch: 30 [32000/50000 (64%)]	Loss: 0.763544

Test set: Average loss: 1.2500, Accuracy: 5681/10000 (57%)

Train Epoch: 31 [0/50000 (0%)]	Loss: 1.428560
Train Epoch: 31 [32000/50000 (64%)]	Loss: 0.866335

Test set: Average loss: 1.0432, Accuracy: 6401/10000 (64%)

Train Epoch: 32 [0/50000 (0%)]	Loss: 0.618042
Train Epoch: 32 [32000/50000 (64%)]	Loss: 0.862750

Test set: Average loss: 1.0184, Accuracy: 6557/10000 (66%)

Train Epoch: 33 [0/50000 (0%)]	Loss: 0.850557
Train Epoch: 33 [32000/50000 (64%)]	Loss: 0.715635

Test set: Average loss: 0.9893, Accuracy: 6694/10000 (67%)

Train Epoch: 34 [0/50000 (0%)]	Loss: 0.861882
Train Epoch: 34 [32000/50000 (64%)]	Loss: 0.549194

Test set: Average loss: 1.0338, Accuracy: 6493/10000 (65%)

Train Epoch: 35 [0/50000 (0%)]	Loss: 0.734958
Train Epoch: 35 [32000/50000 (64%)]	Loss: 0.921221

Test set: Average loss: 1.0233, Accuracy: 6568/10000 (66%)

Train Epoch: 36 [0/50000 (0%)]	Loss: 0.873697
Train Epoch: 36 [32000/50000 (64%)]	Loss: 0.685574

Test set: Average loss: 1.0677, Accuracy: 6400/10000 (64%)

Train Epoch: 37 [0/50000 (0%)]	Loss: 0.739365
Train Epoch: 37 [32000/50000 (64%)]	Loss: 0.762841

Test set: Average loss: 1.3216, Accuracy: 5665/10000 (57%)

Train Epoch: 38 [0/50000 (0%)]	Loss: 1.295749
Train Epoch: 38 [32000/50000 (64%)]	Loss: 1.118892

Test set: Average loss: 1.0372, Accuracy: 6487/10000 (65%)

Train Epoch: 39 [0/50000 (0%)]	Loss: 0.616621
Train Epoch: 39 [32000/50000 (64%)]	Loss: 0.729024

Test set: Average loss: 1.0084, Accuracy: 6597/10000 (66%)

Train Epoch: 40 [0/50000 (0%)]	Loss: 0.779975
Train Epoch: 40 [32000/50000 (64%)]	Loss: 1.066655

Test set: Average loss: 1.0070, Accuracy: 6582/10000 (66%)

Train Epoch: 41 [0/50000 (0%)]	Loss: 0.701568
Train Epoch: 41 [32000/50000 (64%)]	Loss: 0.676124

Test set: Average loss: 1.1028, Accuracy: 6261/10000 (63%)

Train Epoch: 42 [0/50000 (0%)]	Loss: 1.063525
Train Epoch: 42 [32000/50000 (64%)]	Loss: 0.832005

Test set: Average loss: 0.9948, Accuracy: 6648/10000 (66%)

Train Epoch: 43 [0/50000 (0%)]	Loss: 0.801928
Train Epoch: 43 [32000/50000 (64%)]	Loss: 0.553980

Test set: Average loss: 1.2658, Accuracy: 5854/10000 (59%)

Train Epoch: 44 [0/50000 (0%)]	Loss: 1.043638
Train Epoch: 44 [32000/50000 (64%)]	Loss: 0.953023

Test set: Average loss: 1.0059, Accuracy: 6666/10000 (67%)

Train Epoch: 45 [0/50000 (0%)]	Loss: 0.571253
Train Epoch: 45 [32000/50000 (64%)]	Loss: 0.599473

Test set: Average loss: 1.0260, Accuracy: 6617/10000 (66%)

Train Epoch: 46 [0/50000 (0%)]	Loss: 0.559302
Train Epoch: 46 [32000/50000 (64%)]	Loss: 0.647430

Test set: Average loss: 1.0227, Accuracy: 6587/10000 (66%)

Train Epoch: 47 [0/50000 (0%)]	Loss: 0.693760
Train Epoch: 47 [32000/50000 (64%)]	Loss: 0.654179

Test set: Average loss: 1.0364, Accuracy: 6515/10000 (65%)

Train Epoch: 48 [0/50000 (0%)]	Loss: 0.771469
Train Epoch: 48 [32000/50000 (64%)]	Loss: 0.780566

Test set: Average loss: 1.0166, Accuracy: 6608/10000 (66%)

Train Epoch: 49 [0/50000 (0%)]	Loss: 0.606872
Train Epoch: 49 [32000/50000 (64%)]	Loss: 0.713390

Test set: Average loss: 1.0667, Accuracy: 6518/10000 (65%)

Train Epoch: 50 [0/50000 (0%)]	Loss: 0.746445
Train Epoch: 50 [32000/50000 (64%)]	Loss: 0.680349

Test set: Average loss: 1.0220, Accuracy: 6586/10000 (66%)
```




## Visualize STN Results

## Reference

<https://arxiv.org/pdf/1506.02025v3.pdf>
<https://brsoff.github.io/tutorials/intermediate/spatial_transformer_tutorial.html> <https://kevinzakka.github.io/2017/01/10/stn-part1/> <https://kevinzakka.github.io/2017/01/18/stn-part2/> <https://medium.com/@kushagrabh13/spatial-transformer-networks-ebc3cc1da52d>