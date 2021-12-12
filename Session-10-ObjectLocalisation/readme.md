# Object Localization

## TinyImageNet
- Download [TINY IMAGENET](http://cs231n.stanford.edu/tiny-imagenet-200.zip) dataset and train ResNet18 on this dataset (70/30 split) for 50 Epochs with target of 50%+ Validation Accuracy.
- Trained TinyImagenet for 50 epoch with 55.89% validation and 66.07% of training accuracy, details can be found here
- ResNet18 architecture
-  Total params: 11,271,432
-  Trainable params: 11,271,432
-  Non-trainable params: 0
-  Input size (MB): 0.05
-  Forward/backward pass size (MB): 45.00
-  Params size (MB): 43.00
-  Estimated Total Size (MB): 88.05

## Object Detection and Anchor box using K-Means
- Download COCO dataset and learn how COCO object detection dataset's schema is.
- Identify following things for this dataset:
- Class distribution (along with the class names) along with a graph
- Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.
- For coco-dataset optimized k-value for clusters is 4. More details can be found here
