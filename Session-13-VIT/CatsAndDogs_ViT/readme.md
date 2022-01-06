# Vision Transformers with PyTorch
## Objective
The objective of this repo is to train dogs and cats classification dataset using Vision Transformers. We have used two approaches:
- With the blog reference: [Cats&Dogs viT hands on blog](https://analyticsindiamag.com/hands-on-vision-transformers-with-pytorch/). Implemented the code for Vision Transformers with PyTorch using vit_pytorch package and Linformer
- Used transfer learning approach, here we used open-source library [Timm](https://rwightman.github.io/pytorch-image-models/models/vision-transformer/). It is a library of SOTA architectures with pre-trained weights), we picked vit_base_patch16_224 for our training

## Dataset.
Dataset is downloaded from Kaggle [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
- The train folder contains 25000 images of dogs and cats. Each image in this folder has the label as part of the filename. 
- The test folder contains 12500 images, named according to a numeric id. For each image in the test set, you should predict a probability that the image is a dog or cat (1 = dog, 0 = cat)

## VIT Model Training

### Model using vit-pytorch and without Transfer Learning

The notebook can be found from [VIT Pytorch without Transfer Learning](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-13-VIT/CatsAndDogs_ViT/EVA7_Cats_Dogs_ViT.ipynb)

**Model Creation**

```python
efficient_transformer = Linformer(
    dim=768,
    seq_len=196+1,  # 14x14 patches + 1 cls-token
    depth=12,
    heads=12,
    k=64
)

#Visual transformer 
model = ViT(
    dim=768,
    image_size=224,
    patch_size=16,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)

```

**Training log details:**

```
  0%|          | 0/313 [00:00<?, ?it/s]

Epoch : 1 - loss : 0.7196 - acc: 0.5079 - val_loss : 0.6949 - val_acc: 0.4984

  0%|          | 0/313 [00:00<?, ?it/s]

Epoch : 2 - loss : 0.6852 - acc: 0.5483 - val_loss : 0.6748 - val_acc: 0.5732

  0%|          | 0/313 [00:00<?, ?it/s]

Epoch : 3 - loss : 0.6785 - acc: 0.5697 - val_loss : 0.6725 - val_acc: 0.5864

  0%|          | 0/313 [00:00<?, ?it/s]

Epoch : 4 - loss : 0.6743 - acc: 0.5749 - val_loss : 0.6662 - val_acc: 0.6048

  0%|          | 0/313 [00:00<?, ?it/s]

Epoch : 5 - loss : 0.6666 - acc: 0.5917 - val_loss : 0.6550 - val_acc: 0.6159
```

### Model using vit-pytorch and with Transfer Learning

The notebook can be found from [VIT Pytorch with Transfer Learning](https://github.com/amitkml/Transformer-DeepLearning/blob/main/Session-13-VIT/CatsAndDogs_ViT/EVA7_Cats_Dogs_ViT_TransferLearning.ipynb)

**Model Creation**

```python
NUM_FINETUNE_CLASSES = 2
# model = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)vit_base_patch16_224
model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=NUM_FINETUNE_CLASSES).to(device)
```

**Training log details:**

```
HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

Epoch : 1 - loss : 0.0628 - acc: 0.9751 - val_loss : 0.0422 - val_acc: 0.9844

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

Epoch : 2 - loss : 0.0443 - acc: 0.9820 - val_loss : 0.0490 - val_acc: 0.9814

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

Epoch : 3 - loss : 0.0459 - acc: 0.9816 - val_loss : 0.0717 - val_acc: 0.9727

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

Epoch : 4 - loss : 0.0430 - acc: 0.9819 - val_loss : 0.0542 - val_acc: 0.9796

HBox(children=(FloatProgress(value=0.0, max=313.0), HTML(value='')))

Epoch : 5 - loss : 0.0397 - acc: 0.9844 - val_loss : 0.0525 - val_acc: 0.9800
```

**Observation**:

| Scenario                         | Model without Transfer Learning                              | Model with Transfer Learning                                 |
| -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Validation Accuracy on 1st Epoch | Epoch : 1 - loss : 0.7196 - acc: 0.5079 - val_loss : 0.6949 - val_acc: **0.4984** | Epoch : 1 - loss : 0.0628 - acc: 0.9751 - val_loss : 0.0422 - val_acc: **0.9844** |
| Validation Accuracy on 5th Epoch | Epoch : 5 - loss : 0.6666 - acc: 0.5917 - val_loss : 0.6550 - val_acc: **0.6159** | Epoch : 5 - loss : 0.0397 - acc: 0.9844 - val_loss : 0.0525 - val_acc: **0.9800** |

