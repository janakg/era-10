# ERA Session 9 :: CIFAR10

Description: The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Overview

This repository contains the code for training a classifier on CIFAR10 dataset. The classifier is trained using Depthwise Separable Convolution, Dilated Convolution, and GAP.

Folder Structure

├── utils.py # Plotting, training and test functions

├── model.py # Model definition

└── S9.ipynb.py # Main python notebook file with usage scripts

└── README.md # This file

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/janakg/era-s9.git

2. **Open the main Notebook in the Collab**

    ```bash
    https://colab.research.google.com/github/janakg/era-s9/blob/main/S9.ipynb

3. **For local modules, we load the Github repo by cloning or pulling the code**


## Training
Run for 20 epochs with LR of 0.01 and Batch size of 512

1. We are using albumentations for image augmentation
2. We are using Depthwise Separable Convolution for 3-block 
3. We are using Dilated Convolution for all layers
4. GAP



###Metrics
<!-- ![Training Metrics](image.png) -->

This is running in remote