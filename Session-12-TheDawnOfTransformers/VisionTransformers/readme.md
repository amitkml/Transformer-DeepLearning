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

![im](https://amaarora.github.io/images/vit-02.png)

![im](https://amaarora.github.io/images/vit-03.png)

> If we set the the number of `out_channels` to `768`, and both `kernel_size` & `stride` to `16`, then as shown in `fig-3`, once we perform the convolution operation (where the 2-D Convolution has kernel size `3 x 16 x 16`), we can get the **Patch Embeddings** matrix of size `196 x 768` like below:
>
> As can be seen from `above figure`, the `[cls]` token is a vector of size `1 x 768`. We **prepend** it to the **Patch Embeddings**, thus, the updated size of **Patch Embeddings** becomes `197 x 768`.
>
> Next, we add **Positional Embeddings** of size `197 x 768` to the **Patch Embeddings** with `[cls]` token to get **combined embeddings** which are then fed to the `Transformer Encoder`. This is a pretty standard step that comes from the original Transformer paper - [Attention is all you need](https://arxiv.org/abs/1706.03762).
>
> > Note that the Positional Embeddings and `cls` token vector is nothing fancy but rather just a trainable `nn.Parameter` matrix/vector.

- The first step is to break-down the image into patches, 16x16 patches in this case and flatten them. Thus we create `14 x 14` or `196` such patches.

- These patches are projected using a normal linear layer, a Conv2d layer is used for this for performance gain. This is obtained by using a kernel_size and stride equal to the `patch_size`. Intuitively, the convolution operation is applied to each patch individually. So, we have to first apply the conv layer and then flat the resulting images. We pass these patches through a **linear projection layer** to get `1 x 768` long vector representation for each of the image patches. 

  - we had a total of `196` patches and each patch has been represented as a `1 x 768` long vector. Therefore, the total size of the patch embedding matrix is `196 x 768`. You might wonder why is the vector length `768`? Well, `3 x 16 x 16 = 768`. So, we are not really losing any information as step of this process of getting these **patch embeddings**.

- In the **third step**, we take this patch embedding matrix of size `196 x 768` and similar to [BERT](https://arxiv.org/abs/1810.04805), the authors prepend a `[cls]` token to this sequence of embedded patches and then add **Position Embeddings**. The cls token is just a number placed in front of each sequence (of projected patches). 

  - cls_tokens is a torch Parameter randomly initialized, in the forward the method it is copied B (batch) times and prepended before the projected patches using torch.cat

  - the size of the **Patch Embeddings** becomes `197 x 768` after adding the `[cls]` token and also the size of the **Position Embeddings** is `197 x 768`.

    > Why do we add this class token and position embeddings? You will find a 
    > detailed answer in the original Transformer and Bert papers, but to 
    > answer briefly, the `[class]` tokens are added as a special tokens whose outputs from the `Transformer Encoder`
    > serve as the overall image patch representation. And we add the 
    > positional embeddings to retain the positional information of the 
    > patches. The Transformer model on it’s own does not know about the order
    > of the patches unlike CNNs, thus we need to manually inject some 
    > information about the relative or absolute position of the patches.

- For the model to know the original position of the patches, we need to pass the spatial information. In ViT we let the model learn it. The position embedding is just a tensor of shape 1, n_patches + 1(token), hidden_size that is added to the projected patches. In the forward function below, position_embeddings is summed up with the patches (x)

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

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/images/VIT-Embedding.png?raw=true)



## Encoder

The resulting tensor is passeed into a Transformer. In ViT only the Encoder is used, the Transformer encoder module comprises a Multi-Head Self Attention ( MSA ) layer and a Multi-Layer Perceptron (MLP) layer. [Layernorm](https://paperswithcode.com/method/layer-normalization) (**Layer Normalization**) is applied before every block and residual connection after every block.



The encoder combines multiple layers of Transformer Blocks in a sequential manner. The sequence of the operations is as follows -

Input -> TB1 -> TB2 -> .......... -> TBn (n being the number of layers) -> Output.

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/images/VIT-Encoder.png?raw=true)

```python
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
```

As shown in , the Transformer Encoder consists of alternating layers of **Multi-Head Attention** and **MLP** blocks. Also, as shown in , **Layer Norm** is used before every block and residual connections after every block.

The **MLP**, is a Multi-Layer Perceptron block consists of two linear layers and a GELU non-linearity. The outputs from the **MLP** block are again added to the inputs (skip connection) to get the final output from one layer of the **Transformer Encoder**.

A single layer/block of the **Transformer Encoder** can be visualized as below:

![im](https://amaarora.github.io/images/vit-07.png)

Having looked at a single layer inside the **Transformer Encoder**, let’s now zoom out and look at the complete **Transformer Encoder**.



![fig-6 Transformer Encoder](https://amaarora.github.io/images/vit-06.png)							Transformer Encoder

As can be seen from the image above, a single **Transformer Encoder**  consists of 12 layers. The outputs from the first layer are fed to the  second layer, outputs from the second fed to the third until we get the  final outputs from the 12th layer of the **Transformer Encoder** which are then fed to the **MLP Head** to get class predictions. 

## Block

## Attention

## MLP

# References

- [ViT - AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://amaarora.github.io/2021/01/18/ViT.html)