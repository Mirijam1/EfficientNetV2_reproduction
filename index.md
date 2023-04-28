# EfficientNetV2: Smaller Models and Faster Training by Mingxing Tan, Quoc V. Le

## Introduction

## Authors
The authors of this blog post are:
* Nishant Aklecha
* Mirijam Zhang (s.x.zhang@student.tudelft.nl) [4660129]
* Jahson O'Dwyer Wha Binda

## Paper summary

## Reproduction
The EfficientNetV2 model is available on PyTorch [1]. There are three EfficientNetV2 architectures, EfficientNetV2-S, EfficientNetV2-M, and EfficientNetV2-L. For this reproduction project, EfficientNetV2-S is used as it is a smaller model that uses less parameters and will thus be faster. 
PyTorch includes pretrained weights with the model. However, by default, no pre-trained weights are used.

### New data
The new dataset that we chose to train and test is the '25 Indian bird species with 226k images' [2]. The images are resized to 32x32 and 128x128 and trained and tested on both sizes. 

### Reproduced

#### Training and testing with no weights

#### Testing the model with pre-trained weights
For reproduction of the results, the pretrained weights, EfficientNet_V2_S_Weights.DEFAULT, [3] were used. These weights improve upon the results of the original paper by using a modified version of TorchVision’s new training recipe.

Before using the pre-trained models, the images must be preprocessed (resize with right resolution/interpolation, apply inference transforms, rescale the values etc). A preprocessing transforms function is included in the model weight. 

The transform function for EfficientNet_V2_S can be called as follows: `EfficientNet_V2_S_Weights.DEFAULT.transforms()`. The function accepts: PIL.Image, batched (B, C, H, W) and single (C, H, W) image torch.Tensor objects. It performs the following operations: The images are resized to resize_size=[384] using interpolation=InterpolationMode.BILINEAR, followed by a central crop of crop_size=[384]. Finally the values are first rescaled to [0.0, 1.0] and then normalized using mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. [2]



### Hyperparams check

### Ablation study

## References 
1: https://pytorch.org/vision/main/models/efficientnetv2.html
2: https://www.kaggle.com/datasets/arjunbasandrai/25-indian-bird-species-with-226k-images
3: https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html#torchvision.models.EfficientNet_V2_S_Weights
