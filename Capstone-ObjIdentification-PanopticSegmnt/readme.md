# Understanding the architecture for Panoptic Segmentation

**Object Identification**

![IM](https://github.com/facebookresearch/detr/raw/main/.github/DETR.png)

What it is. Unlike traditional computer vision techniques, DETR approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient.
**Panoptic Segmentation**

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Capstone-ObjIdentification-PanopticSegmnt/Resources/arch.png?raw=true)

# Training a custom DeTr

As a part of the Capstone project, I will train a DeTr to give out the Panoptic Segmentation on some classes for which the data was collected by the students.

- In order to predict the Segmentation, the problem needs to be broken into two
  - Object detection model, for this the training dataset needs to have bounding boxes.
  - Segmentation Model, for this the segmentation labels will be used
- However, recent research has suggested that the two step process can be replaced as one where the Bipartite loss can be used directly to predict the segments.

# From where do we take this Encoded Image?

The Encoded image which is of the dimension d x H/32 x w/32 is the output which is fed from the Encoder Network.

The Image which initially passes through the Pre-trained Convolution backbone gets converted into lower resolution followed by projected into a lower dimension value. This value then gets flattened by multiplying the patch width with patch height. This value which is of dimension (Batch Size, Width/32 * height/32, hidden dimension) gets encoded through the encoder layer and becomes the Encoder Hidden state is finally fed into the Decoder layer for classification and bounding box prediction

# What do we do here?

In the Decoder layer the multi-headed attention layer gets the input as Number of images (1 Foreground and Many backgrounds) multiplied by the dimensions.

The Multi-headed attention then calculates the attention score for each object embedding.

# Where is this coming from?

The above step generates the attention mask. Then these attention masks are cleaned using the convolution network that uses the intermediate activation of the backbone As a result we get high resolution maps where each pixel contains a binary logit of belonging to the mask

# Explain these steps?

The above picture shows the panoptic segmentation method of DETR Decoder network.

The Image which initially passes through the Pre-trained Convolution backbone gets converted into lower resolution followed by projected into a lower dimension value. This value then gets flattened by multiplying the patch width with patch height. This value which is of dimension (Batch Size, Width/32 * height/32, hidden dimension) gets encoded through the encoder layer and becomes the Encoder Hidden state is finally fed into the Decoder layer

In the Decoder layer the multi-headed attention layer gets the input as Number of images (1 Foreground and Many backgrounds) multiplied by the dimensions. The Multi-headed attention then calculates the attention score for each object embedding

Then these attention masks are cleaned using the convolution network that uses the intermediate activation of the backbone. As a result we get high resolution maps where each pixel contains a binary logit of belonging to the mask

All the above masks are combined by assigning each pixel to the map with highest logits using simple pixel wise argmax

The output value is quite impressive where the network provides segmentation for each thing and stuff

# References
- https://github.com/facebookresearch/detr
