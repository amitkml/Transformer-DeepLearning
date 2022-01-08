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

## Patch Embedding
First point in the above VIT architecture image above is that, the image is split into patches, below is the source code that creates PatchEmbeddings:

Key points of this class are:

- image size default is assumed as 224x224
- patch size default is assumed as 16x16
- Embedding dimension is assumed as 768
- The to_2tuple function returns image size as 224x224 from 224 and patch size of 16x16 from 16
- num_patches is being populated
  - 224//16= 14
  - 224/16 = 14
  - 16*16 =  256
- Also, if we look at the default valued of embed_dim, it's 768, which means each of our patches will be 786 pixels long. The input image is split into N patches (N = 14 x 14 vectors for ViT-Base) with dimension of 768 embedding vectors by learnable Conv2d (k=16x16) with stride=(16, 16).

```
class PatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.

    """

    def __init__(self, image_size=224, patch_size=16, num_channels=3, embed_dim=768):
        super().__init__()
        image_size = to_2tuple(image_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values):
        batch_size, num_channels, height, width = pixel_values.shape
        # FIXME look at relaxing size constraints
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]})."
            )
        x = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return x
```

## Position and CLS Embeddings

- positional embedding created with patch size of 14x14 =196 and 1 for CLS token. so cls_token shape will be 197x768 whereas 768 is our default embedding size.
- cls_token size is 1x768

```
cls_token = nn.Parameter(torch.zeros(1, 1, 768))
cls_token.shape
>> torch.Size([1, 1, 768])
```

```
position_embedding = nn.Parameter(torch.zeros(1, 14*14 + 1, 768))
position_embedding.shape

>> torch.Size([1, 197, 768])
```

- We first concatenate (prepend) the class tokens to the patch embedding vectors as the 0th vector and then 197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors, this combined embedding is then fed to the transformer encoder.

```
PatchEmbedding (768x196) + CLS_TOKEN (768X1) → Intermediate_Value (768x197)
Positional Embedding (768x197) + Intermediate_Value (768x197) → Combined Embedding (768x197)
```

[CLS] token is a vector of size 1x768, and nn.Parameter makes it a learnable parameter. The position embedding vectors learn distance within the image thus neighboring ones have high similarity.

```
class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, config):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = PatchEmbeddings(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.hidden_size,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        embeddings = self.patch_embeddings(pixel_values)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings
```

