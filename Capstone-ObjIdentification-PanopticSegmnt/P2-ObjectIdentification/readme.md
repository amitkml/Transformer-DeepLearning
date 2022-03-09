# Input Data Preparation

## Input dataset - Concrete defect dataset

A dataset with mask labeling of three major types of concrete surface defects: crack, spalling and exposed rebar, was prepared for training and testing of the model. In this dataset, three open-domain datasets [1-3] are exploited and merged with a bridge inspection image dataset [4] from the Highways Department. To use the dataset, you need comply with the terms and conditions of using the images from the Highways Department, therefore please write a statement and e-mail it to:czhangbd@connect.ust.hk.

For the COCO data format, first of all, there is only a single JSON file for all the annotation in a dataset or one for each split of datasets(Train/Val/Test).

## References
- https://github.com/adilsammar/detr-fine
- [Panoptic Segmentation Explained](https://hasty.ai/blog/panoptic-segmentation-explained)
- [END-to-END object detection (Facebook AI)](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers)
- [Attention is All you Need](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers)
- [The Annotated DETR](https://amaarora.github.io/2021/07/26/annotateddetr.html)
- [Facebook DETR: Transformers dive into the Object Detection World](https://towardsdatascience.com/facebook-detr-transformers-dive-into-the-object-detection-world-39d8422b53fa)
