# Vision Transformers

Objective is to explain Vision Transformers, Transformer-based architectures for Computer Vision Tasks as proposed in the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) .

Transformers have been the de-facto for NLP tasks, and CNN/Resnet-like architectures have been the state of the art for Computer Vision. This paper mainly discusses the strength and versatility of vision transformers, as it kind of approves that they can be used in recognition and can even beat the state-of-the-art CNN.

Following classes from [this](https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py) implementation will be explained block by block:

- Embeddings
- Encoder
- Block
- Attention
- MLP

**The sequence of the operations in VIT is as follows -**

Input -> CreatePatches -> ClassToken, PatchToEmbed , PositionEmbed -> Transformer -> ClassificationHead -> Output

## Embeddings

This class construct the embeddings from patch, position embeddings. The patch size has been set as 16x16.

```python
config.patches = ml_collections.ConfigDict({'size': (16, 16)})
```

After that, we calculate the patch size and no of patch as below.

```python
patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
n_patches = (img_size[0] // 16) * (img_size[1] // 16)
```



## Encoder

## Block

## Attention

## MLP