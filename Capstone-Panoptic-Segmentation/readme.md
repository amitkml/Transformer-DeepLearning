# Panoptic Segmentation

Before going into the details of panoptic segmentation, let us understand two important terminologies germane to image segmentation.

1. **Things** – Any countable object is referred to as a thing in Computer Vision projects. To exemplify – person, cat, car, key, ball are called things.

1. **Stuff** – Uncountable amorphous region of identical texture is known as stuff. For instance, road, water, sky etc.

Study of things falls under the category of instance segmentation task while study of stuff is a semantic segmentation task.

Panoptic segmentation assigns two labels to each of the pixels of an image – (i)semantic label (ii) instance id. The pixels having the same label are considered belonging to the same semantic class and instance id’s differentiate its instances. Unlike instance segmentation, each pixel in panoptic segmentation has a unique label corresponding to instance which means there are no overlapping instances.
![im](https://media-exp1.licdn.com/dms/image/C4E22AQEVj37P_EbaTw/feedshare-shrink_2048_1536/0/1642738925164?e=1645660800&v=beta&t=YSCvbHV-JUo9v7R2i_QjjDjr-RbJqcW93kFZyYdrJUQ)
