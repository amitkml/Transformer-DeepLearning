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
- MLP (Classification) Head: The 0th output from the encoder is fed to the MLP head for classification to output the final classification results.

![im](https://github.com/hirotomusiker/schwert_colab_data_storage/blob/master/images/vit_demo/vit_input.png?raw=true)

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
- A learnable class token is prepended to the patch embedding vectors as the 0th vector.
  197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors.

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

## Transformer Encoder

- N (=197) embedded vectors are fed to the L (=12) series encoders.
- The vectors are divided into query, key and value after expanded by an fc layer.
- q, k and v are further divided into H (=12) and fed to the parallel attention heads.
- Outputs from attention heads are concatenated to form the vectors whose shape is the same as the encoder input.
- The vectors go through an fc, a layer norm and an MLP block that has two fc layers.

The Vision Transformer employs the Transformer Encoder that was proposed in the [attention is all you need paper](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

![im](https://camo.githubusercontent.com/d680008606ecb5d6f0f3dd214781b340c1a3a7ce7a0e325d0a0323f83256dd22/68747470733a2f2f6769746875622e636f6d2f6869726f746f6d7573696b65722f736368776572745f636f6c61625f646174615f73746f726167652f626c6f622f6d61737465722f696d616765732f7669745f64656d6f2f7472616e73666f726d65725f656e636f6465722e706e673f7261773d74727565)

- The configuration values for the ViT model is specified in the sources code under ViTConfig class as shared below:

```
class ViTConfig():
  def __init__(
        self,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=False,
        image_size=224,
        patch_size=16,
        num_channels=3,
        **kwargs
    ):

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels


configuration = ViTConfig()
```

```
Input tensor to Transformer (z0):  torch.Size([1, 197, 768])
Entering the Transformer Encoder 0
Entering the Transformer Encoder 1
Entering the Transformer Encoder 2
Entering the Transformer Encoder 3
Entering the Transformer Encoder 4
Entering the Transformer Encoder 5
Entering the Transformer Encoder 6
Entering the Transformer Encoder 7
Entering the Transformer Encoder 8
Entering the Transformer Encoder 9
Entering the Transformer Encoder 10
Entering the Transformer Encoder 11
Output vector from Transformer (z12-0): torch.Size([1, 768])
```

## ViTSelfAttention

```
Transformer Multi-head Attention block:
Attention(
  (qkv): Linear(in_features=768, out_features=2304, bias=True)
  (attn_drop): Dropout(p=0.0, inplace=False)
  (proj): Linear(in_features=768, out_features=768, bias=True)
  (proj_drop): Dropout(p=0.0, inplace=False)
)
input of the transformer encoder: torch.Size([1, 197, 768])
```

Split qkv into mulitple q, k, and v vectors for multi-head attantion

```
split qkv :  torch.Size([197, 3, 12, 64])
transposed ks:  torch.Size([12, 64, 197])
```

```
class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
```



## MLP (Classification) Head

The 0-th output vector from the transformer output vectors (corresponding to the class token input) is fed to the MLP head to perform the finally classification. This is implemented in the ViTModel() class in the source code.

```
sequence_output = encoder_output[0]
layernorm = nn.LayerNorm(config.hidden_size, eps=0.00001)
sequence_output = layernorm(sequence_output)
# VitPooler
dense = nn.Linear(config.hidden_size, config.hidden_size)
activation = nn.Tanh()
first_token_tensor = sequence_output[:, 0]
pooled_output = dense(first_token_tensor)
pooled_output = activation(pooled_output)

classifier = nn.Linear(config.hidden_size, 100)
logits = classifier(pooled_output)
```

- we take the output from the final transformer encoder, get the 0th vector, which is the prediction vector
- pass it through a layer norm and we take first token out of the vector
- then optionally pass it through a pooler (which is nothing but a dense layer) and add activation as Tanh, pooler layer is used basically to add in more capacity if required
- this pooled output is then sent to the classifier (which is again a linear layer) to get the final output/prediction
