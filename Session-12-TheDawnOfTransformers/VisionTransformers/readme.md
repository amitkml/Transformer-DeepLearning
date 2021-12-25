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

The patch_embeddings gets then populated by conv2d operation where kernel size and stride is being set same as patch size.

```
self.patch_embeddings = Conv2d(in_channels=in_channels,
                               out_channels=config.hidden_size,
                               kernel_size=patch_size,
                               stride=patch_size)
```

The position embedding is being set by nn. Parameter()  which**receives the tensor that is passed into it**, and does not do any initial processing such as uniformization. The size is being set by number if patch and hidden size 768.

```
self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
```

The cls token then set as per hidden size which is set 768. The shape of cls_token is torch.Size([1, 1, 768])

```
self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
```

Let’s now understand how the forward function is being designed.

- The token size is being set based on input shape length.

```
cls_tokens = self.cls_token.expand(B, -1, -1)
```

- Input x then being sent to self.patch embedding to generate embedding.
- Output from self.patch_embeddings then gets flatten
- Transpose is being done by transpose(-1, -2)
- we then concat cls_token with x
- embedding then sets by adding output x with position embedding
- This embedding then passes through dropout.

```
   def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
```



## Encoder

## Block

## Attention

## MLP