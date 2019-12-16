#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:48:55 2019

@author: tom
"""
import numpy as np;
import cv2;
from torchvision import datasets;

def load_mnist_train(train_amount):
    data = datasets.MNIST("../data", train=True, download = True)
    imgs, labels = data.train_data.numpy() , data.train_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_mnist_train: return following shape arrays");
    print("imgs:", return_data[0:train_amount].shape);
    print("labels:", labels[0:train_amount].shape);
    return return_data[0:train_amount]/255.0, labels[0:train_amount];

def load_mnist_test(test_amount):
    data = datasets.MNIST("../data", train=False, download = True)
    imgs, labels = data.test_data.numpy(), data.test_labels.numpy()
    return_data = np.zeros((imgs.shape[0], 32, 32))
    for i in range(imgs.shape[0]):
        return_data[i] = cv2.copyMakeBorder(imgs[i],2,2,2,2,cv2.BORDER_CONSTANT, value=0)
    print("load_mnist_test: return following shape arrays");
    print("imgs:", return_data[0:test_amount].shape);
    print("labels:", labels[0:test_amount].shape);
    return return_data[0:test_amount]/255.0, labels[0:test_amount];

def load_cifar_train(train_amount):
    data = datasets.CIFAR10("../data", train = True, download = True)
    imgs, labels = data.data, np.array(data.targets)
    print("load_cifar_train: return following shape arrays");
    print("imgs:", imgs[0:train_amount].shape);
    print("labels:", labels[0:train_amount].shape);
    return imgs[0:train_amount]/255.0, labels[0:train_amount];

def load_cifar_test(test_amount):
    data = datasets.CIFAR10("../data", train=False, download = True)
    imgs, labels = data.data, np.array(data.targets);
    print("load_cifar_test: return following shape arrays");
    print("imgs:", imgs[0:test_amount].shape);
    print("labels:", labels[0:test_amount].shape);
    return imgs[0:test_amount]/255.0, labels[0:test_amount];

def load_grimace(img_size, num_test):
    train_labels = np.zeros(18*(20-num_test));
    test_labels = np.zeros(18*num_test);
    train_imgs = np.zeros((int(img_size[0]), int(img_size[1]), 3, 18*(20-num_test)));
    test_imgs = np.zeros((int(img_size[0]), int(img_size[1]), 3, 18*num_test));
    for i in range(18):
        path = "./grimace/dataset/" + str(i) + ".npy"
        subdata = np.load(path);
        for j in range(20):
            if j < num_test:
                test_labels[num_test*i+j] = i;
                test_imgs[:, :, :, num_test*i+j] = cv2.resize(subdata[:, :, :, j], (img_size[1], img_size[0]));
            else:
                train_labels[(20-num_test)*i+(j-num_test)] = i
                train_imgs[:, :, :, (20-num_test)*i+(j-num_test)] = cv2.resize(subdata[:, :, :, j], (img_size[1], img_size[0]));
    return np.uint8(train_imgs), train_labels, np.uint8(test_imgs), test_labels;
            