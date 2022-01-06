# Vision Transformers with PyTorch
## Objective
The objective of this repo is to train dogs and cats classification dataset using Vision Transformers. We have used two approaches:
- With the blog reference: [Cats&Dogs viT hands on blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/). Implemented the code for Vision Transformers with PyTorch using vit_pytorch package and Linformer
- Used transfer learning approach, here we used open-source library [Timm](https://rwightman.github.io/pytorch-image-models/models/vision-transformer/). It is a library of SOTA architectures with pre-trained weights), we picked vit_base_patch16_224 for our training

## Dataset.
Dataset is downloaded from Kaggle [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
- The train folder contains 25000 images of dogs and cats. Each image in this folder has the label as part of the filename. 
- The test folder contains 12500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog or cat (1 = dog, 0 = cat)
