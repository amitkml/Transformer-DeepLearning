

# DETR : End-to-End Object Detection with Transformers

Take a look at this [post](https://opensourcelibs.com/lib/finetune-detr), which explains how to fine-tune DETR on a custom dataset.

- The goal is to the process and train the model by own. The objectives are:
  - to understand how fine-tuning works
  - to understand architectural related concepts

Let's first understand Object detection and architectural related concepts for DETR.

## DETR

Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

![im](https://blog.yunfeizhao.com/img/DETR/detailed_detr_architecture.png)

### Transformer architecture

![im](https://blog.yunfeizhao.com/img/DETR/transformer.png)

### Architecture of DETR

![im](https://amaarora.github.io/images/annotated_tfmr_detr.png)

*It contains three main components: a CNN backbone to extract a compact feature representation, an encoder-decoder transformer, and a simple feed forward network (FFN) that makes the final detection prediction.*

- a CNN backbone
- an Encoder-Decoder transformer
- a simple a feed-forward network as a head

The loss is calculated by computing the bipartite matching loss. The model makes a predefined number of predictions, and each of the predictions are computed in parallel.

- backbone upon accepting an input returns `out` and `pos` and here feature size is 2048

- Before the `out` is passed to the transformer, the number of channels are reduced as mentioned in the paper. Here, *d* is set to 256, therefore

  *First, a 1x1 convolution reduces the channel dimension of the high-level activation map ff\*f* from CC\*C* to a smaller dimension dd\*d*. creating a new feature map*

- As also mentioned in the paper,Therefore, we can reshape both feature map and spatial positional encodings to shape

  *The encoder expects a sequence as input, hence we collapse the spatial dimensions of z0 into one dimension, resulting in a d×*H**W feature map.*

**The overall implementation of the DETR architecture has been shown below:**

```python
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    def forward(self, samples: NestedTensor):
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out
```

All the magic happens inside `backbone` and `transformer`, which we will look at next.

### CNN Backbone

Starting from the initial image 
$$
x 
img
​
 ∈R 
3
 ×H 
0
​
 ×W 
0
$$
(with 3 color channels), a conventional CNN backbone generates a lower-resolution activation map. *Typical values we use are C = 2048 and H,W = 
$$
= 
H
0/
32
,
W
0/
32
$$


```python
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
```

Above we create a simple backbone that inherits from `BackboneBase`. The backbone is created using `torchvision.models` and supports all models implemented in `torchvision`. For a complete list of supported models, refer [here](https://pytorch.org/vision/stable/models.html).

As also mentioned above, the typical value for number of channels in the output feature map is 2048, therefore, for all models except `resnet18` & `resnet34`, the `num_channels` variable is set to 2048. This `Backbone` accepts a three channel input image tensor of shape 3×H0×W03×H_0×W_03×*H*0×*W*0, where H0 refers to the input image height, and W0 refers to the input image width.

### Encoder-decoder architecture

First in the DETR model, the image is fed into a CNN such as ResNet-50. A positional encoding, similar to one used for natural language processing, is added to the output and then fed into the transformer system to encode the x and y coordinates of the image.

- There are, however, no words or word embeddings in an image. Instead, each pixel is fed into the transformer like a word, and each pixel’s features act like its respective word embedding. The encoder then acts the same way as it did in natural language processing.

- As for the decoder, it does not make sense to start off its input with a start token. In fact, object detection does not need to detect objects in sequence, so there is no need to continuously concatenate the output to the next input in the decoder. Instead, a fixed number of trainable inputs (in DETR, 100) are used for the decoder called object queries. Each of these object queries can be thought of as a single question on whether an object is in a certain region. This also means that each object query represents how many objects that the model can detect. Each of the decoder’s outputs are then fed into a fully connected network which is used as a classifier to determine what the object is.

- Encoder: Each encoder layer has a standard architecture and consists of a multi-head self-attention module and a feed forward network (FFN).

## Bipartite loss, and why we need it

## Object queries

## Model Training

## Model Results

## References

- [The Annotated DETR](https://amaarora.github.io/2021/07/26/annotateddetr.html)
- [DETR in Visual](https://blog.yunfeizhao.com/2021/04/04/DETR/)
- [Hands-on tutorial for DETR](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=_GQzINI-FBWp)
- [DETR - End to end object detection with transformers (ECCV2020)](https://www.youtube.com/watch?v=utxbUlo9CyY)

