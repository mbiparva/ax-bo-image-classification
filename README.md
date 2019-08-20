# Hyperparameter Bayesian Optimization for Image Classification in PyTorch
This PyTorch implementation demonstrates the Bayesian Optimization (BO) of the hyperparameters involved in the training of a neural network. We try to show a modular implementation of a network in PyTorch that has the capability to optimize automatically over a set of hyperparameters and find the best setting for the final training phase.

## Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Pre-trained Base Networks](#pre-trained-base-networks)
4. [Datasets](#datasets)
5. [Preparation](#preparation)
6. [Training and Testing Procedures](#training-and-testing-procedures)
7. [Ax Parametrization](#ax-parametrization)

## Introduction
Convolutional Neral Networks (ConvNets) not only have network parameters that have to be optimized but also hyperparameters that are commonly manually set according to some domain and prior expert knowledge. There are a number of hyperparameter configuration strategies that are commonly used in the community:
* Grid search
* Random search
* Bayesian Optimization

Bayesian Optimization (BO) has recently gained momentum to systematically find the best hyperparameter settings for a particular experimental setup. We aim to present an attempt that uses the recent libraries in the PyTorch ecosystem to accomplish this task.

### GPyTorch
Gaussian Processes (GP) are non-parametric learning models that use Bayes theorem to do learning and inference originally for regression. It is extended to the classification using proper probability distributions. Kernel machines such as GP also benefits from the notion of feature expansion in an elegant mathematical formulation. [GPyTorch](https://gpytorch.ai/) is an efficient and modular implementation of GPs implemented in PyTorch with the scalability, modularity, and speed criteria in mind.

### BOTorch
Another layer of abstraction sits on top of the GPytorch to further prepare the use of GPs for the Bayesian Optimization task. [BOTorch](https://botorch.org/) is developed to provide a modular and scalable library for [sequential optimization](https://en.wikipedia.org/wiki/Bayesian_optimization) of black-box functions. It is part of automatic machine leaning toolbox that helps to reduce human-driven manual fine-tuning of hyperparameters in neural network training.

### Adaptive Experimentation Platform (Ax)
[Ax](https://ax.dev/) is another layer of abstraction that wraps BOTorch for the ease of development. It is designed to minimize the coding efforts in using BOTorch. Ax has the Service API that lets the user to control scheduling of trials and data prepration and pre-processing pipeline. Ax library, on the other hand, takes care proposing for candidate hyperparameters.

## Installation

1. Clone the ax-bo-image-classification

```shell
# Clone the repository
git clone https://github.com/mbiparva/ax-bo-image-classification.git
```

2. Go into the tools directory

```shell
cd tools
```

3. Run the training or testing script
```shell
# to run the Ax training script
python ax_bo_trainer.py
# to test
python test.py
```

## Pre-trained Base Networks
There is no need to have pre-trained base networks.

## Datasets
This implementation has an option in the configuration file to select one of the following three datasets for the experimentation: MNIST, CIFAR-10, and CIFAR-100. You can easily extend for the other datasets as the code is written as modular as possible to detach the data processing layer form the other layers.

### Directory Hierarchy
The code automatically download the datasets and put them in /dataset directory.

## Preparation
This implementation is tested on the following packages:
* Python 3.7
* PyTorch 1.2
* CUDA 10.1
* EasyDict
* GPyTorch 3.5
* BOTorch 0.1.3
* Ax 0.1.3

## Training and Testing Procedures
In case you used Ax training script to find the best hyperparameter setting, you can then train or test the network by using the "train.py" or "test.pt" as follows.

### Training Script
You can use the tools/train.py to start training the network. If you use --help you will see the list of optional sys arguments that could be passed such as "--use-gpu" and "--gpu-id". You can also have a custom cfg file loaded to customize the reference one if you would not like to change the reference one. Additionally, you can set them one by one once you call "--set".

### Test Script
You can use the tools/test.py to start testing the network by loading a custom network snapshot. You have to pass "--pre-trained-id" and "--pre-trained-epoch" to specify the network id and the epoch the snapshot was taken at.

### Configuration File
All of the configuration hyperparameters are set in the lib/utils/config.py. If you want to change them permanently, simply edit the file with the settings you would like to. Otherwise, use the approaches mentioned above to temporary change them.

## Ax Parametrization
We only demonstrate here the way Ax can be integrated into a training/testing paradigm and develop the layer of abstraction that sits on top of the core implementation. For the sake of simplicity, we show how to use BO to search over three hyperparameters: (1) Learning Rate (lr), (2) Momentum, and (3) weight-decay. If you wish to extend the parametrization, please consult with the documentation of Ax and BOTorch libraries for furhter information.
