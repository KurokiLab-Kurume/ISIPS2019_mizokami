#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:50:44 2019

@author: tom
"""
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sporco import plot

def plot_3d(data, label, offset = False):
    pca = PCA(n_components = 3)
    forplot = pca.fit_transform(data)
    fig = plt.figure(figsize = (10, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("tab10")
    if offset == False:
        for i in range(np.max(label)+1):
            marker = "$" + str(i) + "$"
            index = np.where(label == i)
            x = forplot[index, 0]
            y = forplot[index, 1]
            z = forplot[index, 2]
            ax.scatter(x, y, z, color = cmap(i), marker = marker)
    else:
        for i in range(1, np.max(label)+1):
            marker = "$" + str(i) + "$"
            index = np.where(label == i)
            x = forplot[index, 0]
            y = forplot[index, 1]
            z = forplot[index, 2]
            ax.scatter(x, y, z, color = cmap(i), marker = marker)
        
        plt.show()
    
def plot_1(img, title="img1"):
    fig = plot.figure(figsize=(12, 12));
    plot.imview(img, title = title, fig = fig);
    
def plot_2(imgs, titles = ["img1", "img2"],  filename=""):
    fig = plot.figure(figsize=(12, 6));
    ax1 = fig.add_subplot(121);
    ax2 = fig.add_subplot(122);
    plot.imview(imgs[0], fig=fig, ax=ax1, title=titles[0]);
    plot.imview(imgs[1], fig=fig, ax=ax2, title=titles[1]);
    if filename:
        plot.savefig(filename, dpi = 200, format='png');
        
def plot_3(imgs, titles = ["img1", "img2", "img3"],  filename=""):
    fig = plot.figure(figsize=(18, 6));
    ax1 = fig.add_subplot(131);
    ax2 = fig.add_subplot(132);
    ax3 = fig.add_subplot(133);
    plot.imview(imgs[0], fig=fig, ax=ax1, title=titles[0]);
    plot.imview(imgs[1], fig=fig, ax=ax2, title=titles[1]);
    plot.imview(imgs[2], fig=fig,ax=ax3, title=titles[2]);
    if filename:
        plot.savefig(filename, dpi = 200, format='png');

def plot_4(imgs, titles = ["img1", "img2", "img3", "img4"],  filename=""):
    fig = plot.figure(figsize=(18, 18));
    ax1 = fig.add_subplot(221);
    ax2 = fig.add_subplot(222);
    ax3 = fig.add_subplot(223);
    ax4 = fig.add_subplot(224);
    plot.imview(imgs[0], fig=fig,ax=ax1, title=titles[0]);
    plot.imview(imgs[1], fig=fig,ax=ax2, title=titles[1]);
    plot.imview(imgs[2], fig=fig,ax=ax3, title=titles[2]);
    plot.imview(imgs[3], fig=fig,ax=ax4, title=titles[3]);
    if filename:
        plot.savefig(filename, dpi = 200, format='png');

        
def plot_6(imgs, titles = ["img1", "img2", "img3", "img4", "img5", "img6"], filename=""):
    fig = plot.figure(figsize=(18, 12));
    ax1 = fig.add_subplot(231);
    ax2 = fig.add_subplot(232);
    ax3 = fig.add_subplot(233);
    ax4 = fig.add_subplot(234);
    ax5 = fig.add_subplot(235);
    ax6 = fig.add_subplot(236);
    plot.imview(imgs[0], fig=fig, ax=ax1, title=titles[0]);
    plot.imview(imgs[1], fig=fig, ax=ax2, title=titles[1]);
    plot.imview(imgs[2], fig=fig,ax=ax3, title=titles[2]);
    plot.imview(imgs[3], fig=fig, ax=ax4, title=titles[3]);
    plot.imview(imgs[4], fig=fig, ax=ax5, title=titles[4]);
    plot.imview(imgs[5], fig=fig,ax=ax6, title=titles[5]);
    if filename:
        plot.savefig(filename, dpi = 200, format='png');