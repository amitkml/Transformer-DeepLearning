# Coco dataset  Bounding Box Optimal sizing Assignment

1. Download the coco schema file. Learn how COCO object detection dataset's schema is. This file has the same schema. You'll need to discover what those number are. 
2. Identify these things for this dataset:
   1. readme data for class distribution (along with the class names) along with a graph 
   2. Calculate the Anchor Boxes for k = 3, 4, 5, 6 and draw them.
   3. Share the calculations for both via a notebook uploaded on your GitHub Rep

## COCO file format

At a high level, the COCO format defines exactly how your **annotations** (bounding boxes, object classes, etc) and **image metadata** (like height, width, image sources, etc) are stored on disk. 

The Input to this coco schema analysis is a text file which contains the following

1. Class of Object

2. Image Height

3. Image Width

4. Bounding Box

   a) Top Left X Position

   b) Top Left Y Position

   c) Width of Bounding Box

   d) Height of Bounding Box

```
id: 0, height: 330, width: 1093, bbox:[69, 464, 312, 175],
id: 1, height: 782, width: 439, bbox:[359, 292, 83, 199],
id: 3, height: 645, width: 831, bbox:[297, 312, 267, 167],
id: 34, height: 943, width: 608, bbox:[275, 112, 319, 290],
id: 20, height: 593, width: 857, bbox:[71, 368, 146, 147],
id: 61, height: 587, width: 745, bbox:[177, 463, 68, 302],
id: 16, height: 430, width: 405, bbox:[78, 295, 54, 237],
id: 41, height: 513, width: 1179, bbox:[105, 497, 379, 321],
id: 75, height: 638, width: 842, bbox:[192, 142, 398, 248],
id: 12, height: 746, width: 877, bbox:[345, 441, 370, 181],
id: 5, height: 346, width: 463, bbox:[114, 179, 64, 107],
id: 17, height: 847, width: 821, bbox:[294, 182, 271, 349],
id: 0, height: 645, width: 326, bbox:[242, 12, 96, 346],
id: 60, height: 721, width: 590, bbox:[250, 462, 54, 343],
id: 39, height: 663, width: 663, bbox:[180, 256, 343, 386],
id: 42, height: 1001, width: 663, bbox:[398, 73, 248, 342],
id: 61, height: 469, width: 725, bbox:[221, 99, 282, 109],
id: 75, height: 815, width: 808, bbox:[376, 231, 368, 142],
id: 54, height: 865, width: 1034, bbox:[258, 400, 237, 322],
id: 70, height: 840, width: 730, bbox:[478, 469, 133, 153],
id: 66, height: 728, width: 581, bbox:[98, 247, 134, 314],
id: 14, height: 763, width: 715, bbox:[297, 108, 300, 368],
id: 28, height: 612, width: 770, bbox:[2, 214, 388, 223],
```

