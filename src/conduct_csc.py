#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 11:05:36 2019

@author: tom
"""
from visualize import *;
import numpy as np;
from sporco.dictlrn import cbpdndl;
from sporco.dictlrn import prlcnscdl;
from sporco.admm import cbpdn;
from sporco import util;

def cdl(input_, d_size, lmbda, Iter, visualize = False):
    D0 = np.random.uniform(-1.0, 1.0, d_size);
    opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': Iter,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns');
    d = cbpdndl.ConvBPDNDictLearn(D0, input_, lmbda, opt, dmethod='cns');
    D1 = d.solve();
    print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'));
    if visualize:
        plot_2([util.tiledict(D0.squeeze()), util.tiledict(D1.squeeze())], ["initial dictionary", "learned dictionary"]);
    return d, D1, d.getcoef();

def nn_cdl(input_, d_size, lmbda, Iter, visualize = False):
    D0 = np.random.uniform(0, 1.0, d_size);
    opt = cbpdndl.ConvBPDNDictLearn.Options({'Verbose': True, 'MaxMainIter': Iter,
                            'CBPDN': {'rho': 50.0*lmbda + 0.5, 'NonNegCoef': True},
                            'CCMOD': {'rho': 10.0, 'ZeroMean': True}},
                            dmethod='cns');
    d = cbpdndl.ConvBPDNDictLearn(D0, input_, lmbda, opt, dmethod='cns');
    D1 = d.solve();
    print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'));
    if visualize:
        plot_2([util.tiledict(D0.squeeze()), util.tiledict(D1.squeeze())], ["initial dictionary", "learned dictionary"]);
    return d, D1, d.getcoef();

def par_cdl(input_, d_size, lmbda, Iter, visualize = False):
    D0 = np.random.uniform(-1.0, 1.0, d_size);
    opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options({'Verbose': True,
                        'MaxMainIter': Iter,
                        'CBPDN': {'rho': 50.0*lmbda + 0.5},
                        'CCMOD': {'rho': 1.0, 'ZeroMean': True}})
    d = prlcnscdl.ConvBPDNDictLearn_Consensus(D0, input_, lmbda, opt);
    D1 = d.solve();
    print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'));
    if visualize:
        plot_2([util.tiledict(D0.squeeze()), util.tiledict(D1.squeeze())], ["initial dictionary", "learned dictionary"]);
    return d, D1, d.getcoef();

def par_nn_cdl(input_, d_size, lmbda, Iter, visualize = False):
    D0 = np.random.uniform(-1.0, 1.0, d_size);
    opt = prlcnscdl.ConvBPDNDictLearn_Consensus.Options({'Verbose': True,
                        'MaxMainIter': Iter,
                        'CBPDN': {'rho': 50.0*lmbda + 0.5, 'NonNegCoef': True},
                        'CCMOD': {'rho': 1.0, 'ZeroMean': True}})
    d = prlcnscdl.ConvBPDNDictLearn_Consensus(D0, input_, lmbda, opt);
    D1 = d.solve();
    print("ConvBPDNDictLearn solve time: %.2fs" % d.timer.elapsed('solve'));
    if visualize:
        plot_2([util.tiledict(D0.squeeze()), util.tiledict(D1.squeeze())], ["initial dictionary", "learned dictionary"]);
    return d, D1, d.getcoef();

def csc(input_, D, lmbda, Iter):
    opt = cbpdn.ConvBPDN.Options({'Verbose': True, 'MaxMainIter': Iter,
                              'RelStopTol': 5e-3, 'AuxVarObj': False});
    b = cbpdn.ConvBPDN(D, input_, lmbda, opt);
    X = b.solve();
    return X;
    
    