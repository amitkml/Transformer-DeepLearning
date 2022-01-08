# Vision Transformers (ViT) Code
Here we'll explore and look at the visual transformer source code within the [Timm](https://github.com/rwightman/pytorch-image-models) open source library to get a better intuition of what is going on.

## How the Vision Transformer works in a nutshell?
- The total architecture is called Vision Transformer (ViT in short). Let’s examine it step by step.
- Split an image into patches
- Flatten the patches
- Produce lower-dimensional linear embeddings from the flattened patches
- Add positional embeddings
- Feed the sequence as an input to a standard transformer encoder
- Pretrain the model with image labels (fully supervised on a huge dataset)
- Finetune on the downstream dataset for image classification
![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/images/VIT.png?raw=true)
