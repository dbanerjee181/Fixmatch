# FixMatch Implementation

This repository contains an implementation of FixMatch, a semi-supervised learning algorithm for image classification. It's based on the original FixMatch paper and adapted for use in Google Colab.

## Description

FixMatch is a semi-supervised learning method that leverages a combination of strong augmentations and consistency regularization to improve the performance of image classifiers trained with limited labeled data. It achieves state-of-the-art results on various image classification benchmarks.


![image](https://github.com/user-attachments/assets/d3ada4d8-e846-4e47-9abc-8793510d196b)


This repository provides a modified version of the original FixMatch implementation, with adjustments for better compatibility with Google Colab and specific datasets.

## Usage

**1. Cloning the Repository:**

To clone the repository the following command can be used

-- git clone <path of the repository>


**2. Running the code:**

At present the code is supported for the following datasets

CIFAR10
CIFAR100
SVHN
STL10

The principal scripts that are used are

1. train_modified.py - This is the modified Fixmatch implementation with the augmentation change only
2. train_modified_flexmatch.py - This is the  modified Fixmatch implementation with the augmentation change and the adaptive thresholding.

The example commmand to run them is 

!python /content/Fixmatch/train_modified.py --dataset svhn --num-labeled 1000 --arch wideresnet --lr 0.03 --expand-labels --seed 5 --out results/cifar10_fixmatch@1000.5

!python /content/Fixmatch/train_modified_flexmatch.py --dataset svhn --num-labeled 1000 --arch wideresnet --lr 0.03 --expand-labels --seed 5 --out results/cifar10_flexmatch@1000.5


