# Fine-tune Convolutional Neural Network in Keras with ImageNet Pretrained Models

Deep neural networks has a huge number of parameters, often in the range of millions. Training a Convolutional Neural Network on a small dataset (one that is smaller than the number of parameters) greatly affects the ConvNet’s ability to generalize, often result in overfitting. 

Therefore, in practice, one would fine-tune existing networks that are trained on a large dataset like the ImageNet (1.2M labeled images) by continue training it (i.e. running back-propagation) on the smaller dataset we have. Provided that our dataset is not drastically different in context to the ImageNet dataset, the pretrained model will already have learned features that are relevant to our own classification problem. 

The reason to create this repo is that there are not many online resources that provide sample codes for performing fine-tuning, and that there is not a centralized place where we can easily download ImageNet pretrained models for common ConvNet architectures such as VGG, Inception, ResNet, and DenseNet. This repo serves to fill this gap by providing working examples of performing fine-tuning on Cifar10 dataset with ImageNet pretrained models on popular ConvNet implementations.

## Fine-tuning Techniques

The examples provided in this repo uses the following techniques for fine-tuning:

1. **Truncate the last layer** (softmax layer) of the pretrained network and replace it with our new softmax layer that are relevant to our own problem. For example. if our task is a classification on 10 categories, the new softmax layer of the network will be of 10 categories instead of 1000 categories

2. **Use a smaller learning rate** to train the network. Since we expect the pre-trained weights to be quite good already as compared to randomly initialized weights, we do not want to distort them too quickly and too much. A common practice is to make the initial learning rate 10 times smaller than the one used for scratch training.

3. (Optional) **Freeze the weights of the first few layers** of the pre-trained network. This is because the first few layers capture universal features like curves and edges that are also relevant to our new problem. We want to keep those weights intact. Instead, we will get the network to focus on learning dataset-specific features in the subsequent layers.

## ImageNet Pretrained Models

Network|Theano|Tensorflow
:---:|:---:|:---:
VGG-16 | [model (553 MB)](https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing)| -
VGG-19 | [model (575 MB)](https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view?usp=sharing)| -
GoogLeNet (Inception-V1) | [model (54 MB)](https://drive.google.com/open?id=0B319laiAPjU3RE1maU9MMlh2dnc)| -
Inception-V3 | [model (95 MB)](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/inception_v3_weights_th_dim_ordering_th_kernels.h5)| -
Inception-V4 | [model (172 MB)](https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5)| [model (172 MB)](https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5)
ResNet-50 | [model (103 MB)](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5)| [model (103 MB)](https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5)
ResNet-101 | [model (179 MB)](https://drive.google.com/file/d/0Byy2AcGyEVxfdUV1MHJhelpnSG8/view?usp=sharing)| [model (179 MB)](https://drive.google.com/file/d/0Byy2AcGyEVxfTmRRVmpGWDczaXM/view?usp=sharing)
ResNet-152 | [model (243 MB)](https://drive.google.com/file/d/0Byy2AcGyEVxfZHhUT3lWVWxRN28/view?usp=sharing)| [model (243 MB)](https://drive.google.com/file/d/0Byy2AcGyEVxfeXExMzNNOHpEODg/view?usp=sharing)
DenseNet-121 | [model (32 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ)| [model (32 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc)
DenseNet-169 | [model (56 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfN0d3T1F1MXg0NlU)| [model (56 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfSEc5UC1ROUFJdmM)
DenseNet-161 | [model (112 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfVnlCMlBGTDR3RGs)| [model (112 MB)](https://drive.google.com/open?id=0Byy2AcGyEVxfUDZwVjU2cFNidTA)

## Usage

First, download the ImageNet pretrained weights (in hdf5 format) to the `imagenet_models` folder. The code for fine-tuning on Cifar10 is included in [model_name].py. For example, if you want to fine-tune VGG-16, you can type the following command:

```
python vgg16.py
```

The code will automatically download Cifar10 dataset and performs fine-tuning on VGG-16. Please be aware that it might take some time (up to minutes) for the model to compile and load the ImageNet weights. 

If you wish to perform fine-tuning on your own dataset, you have to replace the module which loads the Cifar10 dataset to load your own dataset.

## Requirements

* Keras 1.2.2
* Theano 0.8.2 or TensorFlow 0.12.0
