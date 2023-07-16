# ERA Session 10 :: CIFAR10

Description: The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Overview

This repository contains the code for training a classifier on CIFAR10 dataset. The classifier is trained using custom ResNet and OneCycle Policy

Folder Structure

├── utils.py # Plotting, training and test functions

├── model.py # Model definition

├── custom_resnet.py # Resnet blocks

└── S10.ipynb.py # Main python notebook file with usage scripts

└── README.md # This file

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/janakg/era-s10.git

2. **Open the main Notebook in the Collab**

    ```bash
    https://colab.research.google.com/github/janakg/era-s10/blob/main/S10.ipynb

3. **For local modules, we load the Github repo by cloning or pulling the code**


## Training
Run for 24 epochs with LR of 0.01 and Batch size of 512
Suggested LR: 2.78E-04

Epoch 24
Train: Loss=0.1014 Batch_id=97 Accuracy=96.28 LR=2.7825772246508313e-05: 100%|██████████| 98/98 [00:10<00:00,  9.39it/s]
Test set: Average loss: 0.2787, Accuracy: 9093/10000 (90.93%)


###Metrics
![Alt log](training-log.png)

