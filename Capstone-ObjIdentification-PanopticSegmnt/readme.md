# Understanding the architecture for Panoptic Segmentation

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Capstone-ObjIdentification-PanopticSegmnt/Resources/arch.png?raw=true)

# Training a custom DeTr

As a part of the Capstone project, I will train a DeTr to give out the Panoptic Segmentation on some classes for which the data was collected by the students.

- In order to predict the Segmentation, the problem needs to be broken into two
  - Object detection model, for this the training dataset needs to have bounding boxes.
  - Segmentation Model, for this the segmentation labels will be used
- However, recent research has suggested that the two step process can be replaced as one where the Bipartite loss can be used directly to predict the segments.

