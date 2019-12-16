#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:54:00 2019

@author: tom
"""

from sklearn import svm;

def train_svm(feature, label):
    clf = svm.SVC(C=5, gamma=0.05);
    clf.fit(feature, label);
    train_result = clf.predict(feature);
    precision = sum(train_result == label)/label.shape[0];
    print('Train precision: ', precision);
    return clf;

def test_svm(clf, feature, label):
    test_result = clf.predict(feature);
    precision = sum(test_result == label)/label.shape[0];
    print('Test precision: ', precision);
    return precision;