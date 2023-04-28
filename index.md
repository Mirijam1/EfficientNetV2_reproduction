# EfficientNetV2: Smaller Models and Faster Training by Mingxing Tan, Quoc V. Le

## Introduction

## Authors
The authors of this blog post are:
* Nishant Aklecha
* Mirijam Zhang (s.x.zhang@student.tudelft.nl) [4660129]
* Jahson O'Dwyer Wha Binda

## Paper summary

## Reproduction

### New data

### Reproduced

### Hyperparams Check

### Ablation Study
When performing abaltion studies, the most interesting components for investigation are the ones that contribute significantly to the model's performance. EfficientNetV2 is a state-of-the-art model that is designed to achieve high accuracy while minimizing computational cost.

To understand the components that contribute most to EfficientNetV2's performance, one might conduct an ablation study to investigate the impact of various architectural components, such as the number of layers or the size of the bottleneck layers.

Specifically, we will be exploring whether the proposed configuration mentioned in the literature can be modified to reduce the size of the model or simplify the architecture. By systematically removing or modifying components of the model, we will be able to determine the impact of each component on the overall performance of the model. This will allow us to gain a deeper understanding of the mechanisms underlying the success of the EfficientNetV2 model and evaluate the relevance of the proposed configuration. Ultimately, the results of this study will provide valuable insights into the design of efficient and effective models for computer vision tasks.

Please refer to the default EfficientNetv2 architecture give below to understand for later ablation modifications:
```
default_architecture = [
    FusedMBConvConfig(1, 3, 1, 24, 24, 2),
    FusedMBConvConfig(4, 3, 2, 24, 48, 4),
    FusedMBConvConfig(4, 3, 2, 48, 64, 4),
    MBConvConfig(4, 3, 2, 64, 128, 6),
    MBConvConfig(6, 3, 1, 128, 160, 9),
    MBConvConfig(6, 3, 2, 160, 256, 15),
]
```

#### Replacing the FusedMBConv layers with MBConv layers
```
modified_architecture = [
    MBConvConfig(1, 3, 1, 24, 24, 2),
    MBConvConfig(4, 3, 2, 24, 48, 4),
    MBConvConfig(4, 3, 2, 48, 64, 4),
    MBConvConfig(4, 3, 2, 64, 128, 6),
    MBConvConfig(6, 3, 1, 128, 160, 9),
    MBConvConfig(6, 3, 2, 160, 256, 15),
]
```
FusedMBConv layers are a key component of the EfficientNetV2 architecture, and are designed to reduce the computational cost of the model while maintaining high accuracy.

By replacing the FusedMBConv layers with standard MBConv layers, we sought to determine the impact of this component on the model's performance. We hypothesized that removing the FusedMBConv layers would increase the computational cost of the model but may also lead to higher accuracy due to the increased expressive power of the standard MBConv layers.

As for the expected outcome of this ablation study, we anticipated that the accuracy of the modified model would decrease slightly due to the increased computational cost. However, we also expected that the modified model would still achieve relatively high accuracy, as the MBConv layers are a powerful building block in deep learning models. Ultimately, the results of this ablation study would provide valuable insights into the relative importance of the FusedMBConv layers and inform the design of future models for computer vision tasks.

#### Replacing the MBConv layers with FusedMBConv layers
```
modified_architecture = [
    FusedMBConvConfig(1, 3, 1, 24, 24, 2),
    FusedMBConvConfig(4, 3, 2, 24, 48, 4),
    FusedMBConvConfig(4, 3, 2, 48, 64, 4),
    FusedMBConvConfig(4, 3, 2, 64, 128, 6),
    FusedMBConvConfig(6, 3, 1, 128, 160, 9),
    FusedMBConvConfig(6, 3, 2, 160, 256, 15),
]
```
To explore the importance of these MBConv layers, we replaced them with FusedMBConv layers, which offer a more computationally efficient alternative. Our aim was to understand how this change impacted the performance of the model, with a particular focus on whether it would affect accuracy due to the reduced expressive power of the FusedMBConv layers.

Our hypothesis was that substituting MBConv layers with FusedMBConv layers would lead to a reduction in the computational cost of the model, but might also cause a decrease in accuracy due to the modified architecture. We anticipated that the model would still achieve relatively high accuracy with the FusedMBConv layers, as they are designed to be more efficient while retaining accuracy.


#### Half the Number of Layers
```
modified_architecture = [
    FusedMBConvConfig(1, 3, 1, 24, 24, 1),
    FusedMBConvConfig(4, 3, 2, 24, 48, 2),
    FusedMBConvConfig(4, 3, 2, 48, 64, 2),
    MBConvConfig(4, 3, 2, 64, 128, 3),
    MBConvConfig(6, 3, 1, 128, 160, 4),
    MBConvConfig(6, 3, 2, 160, 256, 7),
]

```
Our hypothesis was that reducing the number of layers in the small EfficientNetV2 model would lead to a decrease in model complexity and computational cost, but might also result in a reduction in accuracy. However, we also believed that the simplified model might be easier to train and have faster inference times.

As for the expected outcome, we anticipated that the simplified model would have a lower computational cost and be faster to train, but that it might have a slightly lower accuracy compared to the original model due to the reduced number of layers.

Our findings from this ablation study would offer valuable insights into the trade-offs between model complexity, computational cost, training time, and model accuracy in the design of efficient deep learning models for computer vision tasks.

#### Dropped two configs from the model layers
For this ablation study, we decided to modify the default layer configuration of the small EfficientNetV2 model. Specifically, we reduced the number of layers from the default configuration shown below.

We expected that the modified model would have a lower computational cost and faster training and inference times compared to the default model, while maintaining a comparable level of accuracy. Our findings from this ablation study would help in guiding the design of efficient deep learning models for computer vision tasks, and provide insights into the trade-offs between model complexity, computational cost, and model accuracy.
```
modified_architecture = [
    FusedMBConvConfig(1, 3, 1, 24, 48, 2),
    FusedMBConvConfig(4, 3, 2, 48, 64, 4),
    MBConvConfig(4, 3, 2, 64, 128, 6),
    MBConvConfig(6, 3, 2, 128, 256, 15),
]
```

#### Results of Ablation study
Please find accuracy graphs from the three different datasets that we evaluated on:

##### ImageNetTE dataset
![ImageNetTE train](https://github.com/Mirijam1/EfficientNetV2_reproduction/blob/main/images/imagenet_train.png?raw=true)

![ImageNetTE test](https://github.com/Mirijam1/EfficientNetV2_reproduction/blob/main/images/imagenet_test.png?raw=true)

##### MNIST dataset

##### 25-indian-bird-species-with-226k-images dataset
