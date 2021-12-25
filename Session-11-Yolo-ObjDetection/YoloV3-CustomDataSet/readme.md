# YOLO Training on Custom Data

## Data_Collection

The dataset was downloaded from here and additionally 100 images with Creative common license were collected from google for the below classes (25 images per class):
- hardhat
- vest
- boots
- mask

These 100 images where then merged with the above dataset.

## Data_Annotation

Annotation tool from this [repo](https://github.com/miki998/YoloV3_Annotation_Tool) and the installation steps as mentioned in the repo was followed to setup the tool and annotate the images with bounding boxes.

![im](https://user-images.githubusercontent.com/17870236/127248717-cf045180-5342-443c-aada-205b1bb18d9b.png)

```
data
  --customdata
    --images/
      --img001.jpg
      --img002.jpg
      --...
    --labels/
      --img001.txt
      --img002.txt
      --...
    custom.data #data file
    custom.names #class names
    custom.txt #list of name of the images the network to be trained on. Currently we are using same file for test/train
```

## Model_Training

- Created a folder 'weights' in the root (YoloV3) folder and copied the 'yolov3-spp-ultralytics.pt' file downloaded from link
- In 'yolov3-custom.cfg' file,
- Changed the filters as ((4 + 1 + (No of classes))*3)
  - Out no of class is 4
  - So no of filters will be : (4+1+4)*3 = 27
- Changed all entries of classes as 4 (Search for 'classes=80' and change all three entries to 'classes=1')
- Changed burn_in to 100
- Changed max_batches to 5000
- Changed steps to 4000,4500
- Changed the class names in custom.names
  - hardhat
  - vest
  - mask
  - boots
- Visualization of Batch images - Image Augmentation with YOLO - Bag of Freebies

### Logs

```python
oday's Training start date-time: 2021-12-25
Namespace(accumulate=4, adam=False, batch_size=10, bucket='', cache_images=True, cfg='cfg/yolov3-custom.cfg', data='data/customdata/custom.data', device='', epochs=50, evolve=False, img_size=[512], multi_scale=False, name='', nosave=False, notest=False, rect=False, resume=False, single_cls=False, weights='weights/yolov3-spp-ultralytics.pt')
Using CUDA device0 _CudaDeviceProperties(name='Tesla K80', total_memory=11441MB)

Run 'tensorboard --logdir=runs' to view tensorboard at http://localhost:6006/
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
WARNING: smart bias initialization failure.
Model Summary: 225 layers, 6.25895e+07 parameters, 6.25895e+07 gradients
Caching labels (2863 found, 131 missing, 38 empty, 0 duplicate, for 3032 images): 100% 3032/3032 [00:00<00:00, 4566.67it/s]
Caching images (1.7GB): 100% 3032/3032 [00:22<00:00, 137.44it/s]
Reading image shapes: 100% 319/319 [00:00<00:00, 4224.48it/s]
Caching labels (297 found, 15 missing, 7 empty, 0 duplicate, for 319 images): 100% 319/319 [00:00<00:00, 4560.11it/s]
Caching images (0.1GB): 100% 319/319 [00:03<00:00, 104.19it/s]
Image sizes 512 - 512 train, 512 test
Using 2 dataloader workers
Starting training for 50 epochs...

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/304 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/cuda/memory.py:386: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  FutureWarning)
      0/49     7.37G      7.98       155      4.31       167        76       512:   0% 1/304 [00:06<33:19,  6.60s/it]/usr/local/lib/python3.7/dist-packages/torch/cuda/memory.py:386: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  FutureWarning)
      0/49      7.4G      4.83      7.18      2.61      14.6        13       512: 100% 304/304 [19:46<00:00,  3.90s/it]
/usr/local/lib/python3.7/dist-packages/torch/functional.py:445: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2157.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:28<00:00,  1.13it/s]
                 all       319  1.53e+03     0.234     0.424     0.243     0.295

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/304 [00:00<?, ?it/s]/usr/local/lib/python3.7/dist-packages/torch/cuda/memory.py:386: FutureWarning: torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved
  FutureWarning)
      1/49      7.4G       3.8      1.91      1.16      6.87         6       512: 100% 304/304 [19:43<00:00,  3.89s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:27<00:00,  1.18it/s]
                 all       319  1.53e+03     0.385     0.607     0.414     0.458

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      2/49      7.4G      3.36      1.71     0.879      5.95         7       512: 100% 304/304 [19:42<00:00,  3.89s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:27<00:00,  1.18it/s]
                 all       319  1.53e+03     0.435     0.606      0.48     0.506

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
  0% 0/304 [00:00<?, ?it/s]
Model Bias Summary:    layer        regression        objectness    classification
                          89      -0.17+/-0.20      -7.64+/-0.92      -0.66+/-0.48 
                         101       0.04+/-0.22      -7.82+/-1.00      -0.66+/-0.21 
                         113       0.14+/-0.21      -9.32+/-0.66      -0.61+/-0.33 
      3/49      7.4G      3.02      1.62     0.768      5.41        17       512: 100% 304/304 [19:43<00:00,  3.89s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:27<00:00,  1.18it/s]
                 all       319  1.53e+03     0.499     0.639     0.506     0.559

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      4/49      7.4G      2.89       1.6     0.763      5.25        21       512: 100% 304/304 [19:41<00:00,  3.89s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:27<00:00,  1.18it/s]
                 all       319  1.53e+03     0.465     0.645     0.502      0.54

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      5/49      7.4G      2.73       1.5     0.656      4.89         7       512: 100% 304/304 [19:38<00:00,  3.88s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.548     0.651     0.542     0.589

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      6/49      7.4G      2.68      1.44     0.647      4.77        15       512: 100% 304/304 [19:38<00:00,  3.88s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.498     0.629     0.523     0.556

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      7/49      7.4G       2.6      1.38     0.614      4.59        17       512: 100% 304/304 [19:39<00:00,  3.88s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.519     0.667     0.537     0.583

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      8/49      7.4G      2.54      1.37     0.508      4.42        20       512: 100% 304/304 [19:39<00:00,  3.88s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.546     0.623     0.532     0.578

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
      9/49      7.4G      2.48      1.33     0.507      4.31         7       512: 100% 304/304 [19:43<00:00,  3.89s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.546     0.628     0.527     0.583

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     10/49      7.4G      2.45      1.36     0.496      4.31         4       512: 100% 304/304 [19:44<00:00,  3.90s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.543     0.638     0.525     0.584

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     11/49      7.4G      2.41      1.31     0.455      4.17         8       512: 100% 304/304 [19:41<00:00,  3.88s/it]
               Class    Images   Targets         P         R   mAP@0.5        F1: 100% 32/32 [00:26<00:00,  1.19it/s]
                 all       319  1.53e+03     0.559     0.639     0.538     0.595

     Epoch   gpu_mem      GIoU       obj       cls     total   targets  img_size
     12/49      7.4G      2.38      1.29     0.438      4.11        66       512:  83% 252/304 [16:25<03:23,  3.91s/it]
Traceback (most recent call last):
  File "train.py", line 430, in <module>
    train()  # train normally
  File "train.py", line 284, in train
    optimizer.zero_grad()
  File "/usr/local/lib/python3.7/dist-packages/torch/optim/optimizer.py", line 217, in zero_grad
    p.grad.zero_()
KeyboardInterrupt
Today's Training End date-time: 2021-12-25
```

## Model_Inference

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-11-Yolo-ObjDetection/YoloV3-CustomDataSet/YoloV3/output/boots.jpg?raw=true)

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-11-Yolo-ObjDetection/YoloV3-CustomDataSet/YoloV3/output/image-06.jpg?raw=true)

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-11-Yolo-ObjDetection/YoloV3-CustomDataSet/YoloV3/output/image-12.jpg?raw=true)

![im](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-11-Yolo-ObjDetection/YoloV3-CustomDataSet/YoloV3/output/image-13.jpg?raw=true)



### Inference on Youtube video

Video being uploaded into [Yolo Object Detection](https://youtu.be/0KVU5Bs-bHY)