# mnist_cudnn
MNIST(National Institute of Standards and Technology database) handwriting recognition using Deep learning and CUDNN library under ROS 2.

## Built with

- ROS Foxy under Ubuntu 20.04 LTS

- ROS Jazzy under Ubuntu 24.04 LTS

## Required Package

- OpenCV - https://opencv.org/

- NVIDIA CUDA - https://developer.nvidia.com/cuda-toolkit

- NVIDIA CUBLAS - https://developer.nvidia.com/cublas

- NVIDIA CUDNN - https://developer.nvidia.com/cudnn

------

## Getting Started

### Database

ImageProcessing - https://github.com/chankeh/ImageProcessing

------

### Installation

``` bash
$ git clone https://github.com/qaz9517532846/mnist_cudnn.git
```

### Run

------

Train Data

``` bash
MNIST mnist;
mnist.Train("<Train path>", "<Train dataset>", batchSizeTrain, true);
```

``` bash
$ ros2 run mnist_cudnn mnist_cnn_train
```

``` bash
.. model Configuration ..
CUDA: conv1
CUDA: pool
CUDA: conv2
CUDA: pool
CUDA: dense1
CUDA: relu
CUDA: dense2
CUDA: softmax
conv1: Available Algorithm Count [FWD]: 10
conv1: Available Algorithm Count [BWD-filter]: 9
conv1: Available Algorithm Count [BWD-data]: 8
.. initialized conv1 layer ..
conv2: Available Algorithm Count [FWD]: 10
conv2: Available Algorithm Count [BWD-filter]: 9
conv2: Available Algorithm Count [BWD-data]: 8
.. initialized conv2 layer ..
.. initialized dense1 layer ..
.. initialized dense2 layer ..
step:  200, loss: 0.401, accuracy: 74.867%
step:  400, loss: 0.315, accuracy: 91.447%
step:  600, loss: 0.216, accuracy: 93.490%
step:  800, loss: 0.143, accuracy: 94.633%
step: 1000, loss: 0.123, accuracy: 95.438%
step: 1200, loss: 0.165, accuracy: 95.941%
step: 1400, loss: 0.112, accuracy: 96.439%
step: 1600, loss: 0.085, accuracy: 96.744%
step: 1800, loss: 0.072, accuracy: 96.895%
step: 2000, loss: 0.101, accuracy: 97.061%
step: 2200, loss: 0.079, accuracy: 97.254%
step: 2400, loss: 0.062, accuracy: 97.354%
.. store weights to the storage ..
.. saving conv1 parameter .. done ..
.. saving pool parameter .. done ..
.. saving conv2 parameter .. done ..
.. saving pool parameter .. done ..
.. saving dense1 parameter .. done ..
.. saving relu parameter .. done ..
.. saving dense2 parameter .. done ..
.. saving softmax parameter .. done ..
```

Test Image

``` bash
MNIST mnist;
mnist.Test("<image file>");
```

``` bash
$ ros2 run mnist_cudnn mnist_cnn_test
```
``` bash
[INFERENCE]
Data Finished
Data Tensor
Data target
Test Finished.. model Configuration ..
CUDA: conv1
CUDA: pool
CUDA: conv2
CUDA: pool
CUDA: dense1
CUDA: relu
CUDA: dense2
CUDA: softmax
conv1: Available Algorithm Count [FWD]: 10
conv1: Available Algorithm Count [BWD-filter]: 9
conv1: Available Algorithm Count [BWD-data]: 8
.. loaded conv1 pretrain parameter..
conv2: Available Algorithm Count [FWD]: 10
conv2: Available Algorithm Count [BWD-filter]: 9
conv2: Available Algorithm Count [BWD-data]: 8
.. loaded conv2 pretrain parameter..
.. loaded dense1 pretrain parameter..
.. loaded dense2 pretrain parameter..
Result = 3
```

------

## Reference:

[1]. mnist_cudnn - https://github.com/haanjack/mnist-cudnn

[2]. cudnn-rnn-check - https://github.com/jzhang533/cudnn-rnn-check/tree/master

------

## License:

This repository is for your reference only. copying, patent applications, and academic journals are strictly prohibited.

Copyright © 2025 ZM Robotics Software Laboratory.
