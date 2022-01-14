

# DETR : End-to-End Object Detection with Transformers

Take a look at this [post](https://opensourcelibs.com/lib/finetune-detr), which explains how to fine-tune DETR on a custom dataset.

- The goal is to the process and train the model by own. The objectives are:
  - to understand how fine-tuning works
  - to understand architectural related concepts

Let's first understand Object detection and architectural related concepts for DETR.

## DETR

Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.

- An  image is sent through a pre-trained convolutional backbone (in the paper, the authors use ResNet-50/ResNet-101). Let’s assume we also add a batch dimension. This means that the input to the backbone is a tensor of shape `(batch_size, 3, height, width)`, assuming the image has 3 color channels (RGB).
- The CNN backbone outputs a new lower-resolution feature map, typically of shape `(batch_size, 2048, height/32, width/32)`.
- This is then projected to match the hidden dimension of the Transformer of DETR, which is `256` by default, using a `nn.Conv2D` layer. So now, we have a tensor of shape `(batch_size, 256, height/32, width/32)`
- Next, the feature map is flattened and transposed to obtain a tensor of shape `(batch_size, seq_len, d_model)` = `(batch_size, width/32*height/32, 256)`. So a difference with NLP models is that the sequence length is actually longer than usual, but with a smaller `d_model` (which in NLP is typically 768 or higher).
- This is sent through the encoder, outputting `encoder_hidden_states` of the same shape (you can consider these as image features). 
- so-called **object queries** are sent through the decoder. This is a tensor of shape `(batch_size, num_queries, d_model)`, with `num_queries` typically set to 100 and initialized with zeros.
- Next, two heads are added on top for object detection: a linear layer for classifying each object query into one of the objects or “no object”, and a MLP to predict bounding boxes for each query.

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

## Bipartite loss and why?

- The model is trained using a **bipartite matching loss**: so what we actually do is compare the predicted classes + bounding boxes of each of the N = 100 object queries to the ground truth annotations, padded up to the same length N (so if an image only contains 4 objects, 96 annotations will just have a “no object” as class and “no bounding box” as bounding box). 

- The [Hungarian matching algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm) is used to find an optimal one-to-one mapping of each of the N queries to each of the N annotations. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and [generalized IoU loss](https://giou.stanford.edu/) (for the bounding boxes) are used to optimize the parameters of the model.

Unlike any other Neural network, we also need a loss function for the DETR network as well. But this loss function is very unique by itself. It’s called Bipartite matching loss. Let’s take an example, let’s say we have the image of a kid’s room which contains many items and our job is to identify each object and the location of the object using bounding boxes. TheDETR network outputs the objects and the bounding boxes as well.

1. But does the network predict the centroid of the objects accurately?
2. Does the network predict properly when multiple objects of the same class present (say one small and one big)?
3. How it predicts when nothing is there in the location
4. If it can’t predict the location accurately then is the loss going to be very huge and the network can’t learn the centroid s accurately at all?

This is where the Bipartite loss comes handy. It performs the updates as follows

1. The loss takes into account the type of object and not the location
2. The loss takes into account the number of objects
3. The loss tries to find out the object which has lowest distance/error

That means let’s say in the room there are 2 cricket bats, 1 foot ball then as long as the network predicts both cricket bats and ball, the job is almost over. The network then needs to match the cricket bats of Ground truth to the prediction which has the lowest distance
Now let’s say there are no hockey bats and network predict a hocket bat then it has to go and correct itself as it will get a high loss Similarly, when there are no objects in GT and same is predicted by Network then no loss is logged. We can see the mathematics of loss function below and it should be easily understood

[![img](https://github.com/nkanungo/EVA6/raw/main/DETR/images/loss.PNG)](https://github.com/nkanungo/EVA6/blob/main/DETR/images/loss.PNG)

## Object queries

![im](https://www.guinnessworldrecords.com/Images/lionel-messi-pes-2020_tcm25-598129.jpg)

What are different questions that comes to your mind? let’s say

1. Who is the one with the ball now?
2. Can you identify the player with red shirt and first from the left?
3. Who is the Midfielder with Blue T-shirt?
4. Which country the first player from left belongs to?
5. Is it an offsite?

Now imagine you are performing the Image identification, Object Detection, Image segmentation kind of solution here using DETR Encoder/Decoder Architecture. The Object queries which are input to the Decoder Self attention layer are type of queries which the network tries to address using the output from the Encoder layer and Attention layer followed by FFN of the Decoder network.

- These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the encoder, they are added to the input of each attention layer.
- Each object query will look for a particular object in the image.
- The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output `decoder_hidden_states` of the same shape: `(batch_size, num_queries, d_model)`. 

## Model Training

## Model Results

## References

- [The Annotated DETR](https://amaarora.github.io/2021/07/26/annotateddetr.html)
- [DETR in Visual](https://blog.yunfeizhao.com/2021/04/04/DETR/)
- [Hands-on tutorial for DETR](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_attention.ipynb#scrollTo=_GQzINI-FBWp)
- [DETR - End to end object detection with transformers (ECCV2020)](https://www.youtube.com/watch?v=utxbUlo9CyY)
- [DETR](https://huggingface.co/docs/transformers/model_doc/detr)
- [DETR: End-to-End Object Detection with Transformers (Paper Explained)](https://www.youtube.com/watch?v=T35ba_VXkMY)

