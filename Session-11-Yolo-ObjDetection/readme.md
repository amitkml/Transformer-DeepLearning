# Object Detection using YOLO v3

## Assignment-A OpenCV Yolo:

**Aim**
- Detecting objects in an image where there is a person and an object present in the image.
- Note: The object should be present in the COCO classes.

**Steps:**
- Followed the steps listed [here](https://pysource.com/2019/06/27/yolo-object-detection-using-opencv-with-python/)
- Took a image with object which is there in COCO data set (search for COCO classes to learn).
- Ran this image through the code above
- Uploaded the annotated image by YOLO OpenCV.

## Assignment-B Yolov3 Object Detection on Custom Dataset

The assignment aim to perform object detection on custom dataset using Yolov3. Custom data included 4 classes : Hardhat, vest, mask, boots.
- Refer to [this](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS) Colab File
- Refer to [this](https://github.com/theschoolofai/YoloV3) GitHub Repo
- Download [this](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view) dataset
- Collect and add 25 images for the following 4 classes into the dataset shared:
-   class names are hardhat, vest, mask and boots, its also listed in custom.names file
-   follow exact rules to make sure that you can train the model. Steps are explained in the README.md file on github repo link above.
-   Once additional 100 images are added, train the model

Next
- Download a very small (~10-30sec) video from youtube which shows the 4 classes.
- Use ffmpeg to extract frames from the video.
- Infer on these images using detect.py file.
- python detect.py --conf-three 0.3 --output output_folder_name
- Use ffmpeg to convert the files in your output folder to video
- Upload the video to YouTube.


Result
Click here to play the output video

## Assignment C - Adding new dataset

### Custom Dataset Preparation

- Image Annotation:  Annotate [the](https://github.com/miki998/YoloV3_Annotation_Tool) custom data using this annotation tool.

### Model Training

- Downloading pretrained weights: Download the file named yolov3-spp-ultralytics.pt from [here](https://drive.google.com/open?id=1LezFG5g3BCW6iYaV89B2i64cqEUZD7e0) and place it in weights directory.
