# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:58:06 2018

@author: Administrator

from factor import scoreF


"""
import numpy as np

def scoreF(data,ndiv):
    '''
    example:
        data = np.random.rand(10,5)
        scoreF(data,5)
        data = np.array([1,7,np.nan,10,6,np.nan,7,4])
        scoreF(data,3)
    '''
    if len(data.shape)==2 :
        SCORES = np.zeros(data.shape)
        for col in range(data.shape[1]):
            d1 = data[:,col]
            non_nan_idx = ~np.isnan(d1)
            value = d1[non_nan_idx]
            C     = np.count_nonzero(non_nan_idx)  
            N     = np.ceil(C/ndiv)
            order = value.argsort()+1
            score = np.ceil(order/N)
            SCORES[non_nan_idx,col]=score
    else:
        SCORES = np.zeros(len(data))
        non_nan_idx = ~np.isnan(data)
        value = data[non_nan_idx]
        C     = np.count_nonzero(non_nan_idx)  
        N     = np.ceil(C/ndiv)
        order = value.argsort()+1
        SCORES[non_nan_idx] = np.ceil(order/N)
    return SCORES