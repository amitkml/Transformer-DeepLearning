# Train DETR for object detection on custom data

In Computer Vision, object detection is a task where we want our model to distinguish the foreground objects from the background and predict the locations and the categories for the objects present in the image.

There are many frameworks out there for object detection but the researchers at Facebook AI has come up with DETR, an innovative and efficient approach to solve the object detection problem.

DETR treats an object detection problem as a direct set prediction problem with the help of an encoder-decoder architecture based on transformers. By set, I mean the set of bounding boxes. 

Transformers are the new breed of deep learning models that have performed outstandingly in the NLP domain. This is the first time when someone used transformers for object detection.

The authors of this paper have evaluated DETR on one of the most popular object detection datasets, COCO, against a very competitive Faster R-CNN baseline.

In the results, the DETR achieved comparable performances. More precisely, DETR demonstrates significantly better performance on large objects. However, it didn’t perform that well on small objects.

The Defacto standard for training any object detection model is using COCO format. To train our model for object detection task we have to prepare our dataset in standard coco format

##  1. <a name='COCOFormatforObjectdetection'></a>COCO Format for Object detection

Microsoft's Common Objects in Context dataset (COCO) is the most popular object detection dataset at the moment. It is widely used to benchmark the performance of computer vision methods.

Due to the popularity of the dataset, the format that COCO uses to store annotations is often the go-to format when creating a new custom object detection dataset. While the COCO dataset also supports annotations for other tasks like segmentation, I will leave that to a future blog post. For now, we will focus only on object detection data.

The “COCO format” is a specific JSON structure dictating how labels and metadata are saved for an image dataset.

###  1.1. <a name='COCOfileformat'></a>COCO file format

If you are new to the object detection space and are tasked with creating a new object detection dataset, then following the [COCO format](https://cocodataset.org/#format-data) is a good choice due to its relative simplicity and widespread usage. This section will explain what the file and folder structure of a COCO formatted object detection dataset actually looks like.
At a high level, the COCO format defines exactly how your annotations (bounding boxes, object classes, etc) and image metadata (like height, width, image sources, etc) are stored on disk.

###  1.2. <a name='FolderStructure'></a>Folder Structure

The folder structure of a COCO dataset looks like this:

    <dataset_dir>/
        data/
            <filename0>.<ext>
            <filename1>.<ext>
            ...
        labels.json


The dataset is stored in a directory containing your raw image data and a single json file that contains all of the annotations, metadata, categories, and other information that you could possibly want to store about your dataset. If you have multiple splits of data, they would be stored in different directories with different json files.

###  1.3. <a name='JSONformat'></a>JSON format

If you were to download the [COCO dataset from their website](https://cocodataset.org/#download), this would be the `instances_train2017.json` and `instances_val2017.json` files.

    {
        "info": {
            "year": "2021",
            "version": "1.0",
            "description": "Exported from FiftyOne",
            "contributor": "Voxel51",
            "url": "https://fiftyone.ai",
            "date_created": "2021-01-19T09:48:27"
        },
        "licenses": [
            {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike License"
            },
            ...   
        ],
        "categories": [
            ...
            {
                "id": 2,
                "name": "cat",
                "supercategory": "animal"
            },
            ...
        ],
        "images": [
            {
                "id": 0,
                "license": 1,
                "file_name": "<filename0>.<ext>",
                "height": 480,
                "width": 640,
                "date_captured": null
            },
            ...
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 2,
                "bbox": [260, 177, 231, 199],
                "segmentation": [...],
                "area": 45969,
                "iscrowd": 0
            },
            ...
        ]
    }


* **Info** — Description and versioning information about your dataset.

* **Licenses** — List of licenses with unique IDs to be specified by your images.

* **Categories** — Classification categories each with a unique ID. Optionally associated with a supercategory that can span multiple classes. These categories can be whatever you want, but note that if you’d need to follow the COCO classes if you want to use a model pretrained on COCO out of the box (or follow other dataset categories to use other models).

* **Images** — List of images in your dataset and relevant metadata including unique image ID, filepath, height, width, and optional attributes like license, URL, date captured, etc.

* **Annotations** — List of annotations each with a unique ID and the image ID it relates to. This is where you will store the bounding box information in our case or segmentation/keypoint/other label information for other tasks. This also stores bounding box area and iscrowd indicating a large bounding box surrounding multiple objects of the same category which is used for evaluation.

2. ## <a name='Different annotations formats'></a>Different annotations formats

Bounding boxes are rectangles that mark objects on an image. There are multiple formats of bounding boxes annotations. Each format uses its specific representation of bouning boxes coordinates. Albumentations supports four formats: `pascal_voc`, `albumentations`, `coco`, and `yolo` .

Let's take a look at each of those formats and how they represent coordinates of bounding boxes.

As an example, we will use an image from the dataset named [Common Objects in Context](http://cocodataset.org/). It contains one bounding box that marks a cat. The image width is 640 pixels, and its height is 480 pixels. The width of the bounding box is 322 pixels, and its height is 117 pixels.

The bounding box has the following `(x, y)` coordinates of its corners: top-left is `(x_min, y_min)` or `(98px, 345px)`, top-right is `(x_max, y_min)` or `(420px, 345px)`, bottom-left is `(x_min, y_max)` or `(98px, 462px)`, bottom-right is `(x_max, y_max)` or `(420px, 462px)`. As you see, coordinates of the bounding box's corners are calculated with respect to the top-left corner of the image which has `(x, y)` coordinates `(0, 0)`.

![im](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_example.jpg)

![im](https://albumentations.ai/docs/images/getting_started/augmenting_bboxes/bbox_formats.jpg)

### 2.1. <a name='pascal_voc'></a>pascal_voc

`pascal_voc` is a format used by the [Pascal VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). Coordinates of a bounding box are encoded with four values in pixels: `[x_min, y_min, x_max, y_max]`. `x_min` and `y_min` are coordinates of the top-left corner of the bounding box. `x_max` and `y_max` are coordinates of bottom-right corner of the bounding box.

Coordinates of the example bounding box in this format are `[98, 345, 420, 462]`.



### 2.2. <a name='coco'></a>coco

`coco` is a format used by the [Common Objects in Context COCOCOCO](http://cocodataset.org/) dataset.

In `coco`, a bounding box is defined by four values in pixels `[x_min, y_min, width, height]`. They are coordinates of the top-left corner along with the width and height of the bounding box.

Coordinates of the example bounding box in this format are `[98, 345, 322, 117]`.

##  3. <a name='CreatingaCustomCOCOformatdataset'></a>Creating a Custom COCO format dataset

###  3.1. <a name='Background'></a>Background

A dataset with mask labeling of three major types of concrete surface defects: crack, spalling and exposed rebar, was prepared for training and testing of the model. In this dataset, three open-domain datasets [1-3] are exploited and merged with a bridge inspection image dataset [4] from the Highways Department. To use the dataset, you need comply with the terms and conditions of using the images from the Highways Department, therefore please write a statement and e-mail it to:czhangbd@connect.ust.hk.
The dataset can be downloaded from [concrete defect dataset](https://drive.google.com/file/d/1UbAnTFQWShtuHlGEvYYZ4TP8tL49IM8t/view?usp=sharing)



## 4. <a name='Generating Bounding Boxes from Mask'></a>Generating Bounding Boxes from Mask

Our input dataset has masks for each of the images and some of them.

I have shown here a sample input image **00001.jpg**

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Capstone-ObjIdentification-PanopticSegmnt/assets/00001.jpg?raw=true)

and we have following masks for the above image. Mask file name is 00001rebar.jpg. So my understanding is that object name here is rebar which is being shown as masked one.





![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Capstone-ObjIdentification-PanopticSegmnt/assets/00001rebar.jpg?raw=true)



## References

- [Training DETR on Your Own Dataset]([https://towardsdatascience.com/training-detr-on-your-own-dataset-bcee0be05522)

- [E2E Object Detection using DETR](https://www.kaggle.com/tanulsingh077/end-to-end-object-detection-with-transformers-detr/notebook)
- [Bounding boxes augmentation for object detection](https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#bounding-boxes-augmentation-for-object-detection)
- [Finetune DETR](https://colab.research.google.com/github/woctezuma/finetune-detr/blob/master/finetune_detr.ipynb)
- https://github.com/adilsammar/detr-fine
- [Panoptic Segmentation Explained](https://hasty.ai/blog/panoptic-segmentation-explained)
- [END-to-END object detection (Facebook AI)](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers)
- [Attention is All you Need](https://ai.facebook.com/blog/end-to-end-object-detection-with-transformers)
- [The Annotated DETR](https://amaarora.github.io/2021/07/26/annotateddetr.html)
- [Facebook DETR: Transformers dive into the Object Detection World](https://towardsdatascience.com/facebook-detr-transformers-dive-into-the-object-detection-world-39d8422b53fa)

