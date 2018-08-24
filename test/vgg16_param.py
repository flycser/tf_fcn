#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : vgg16_param.py
# @Date : 08/21/2018 21:45:20
# @Poject : tf_fcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc : check vgg16 parameters

import pickle

import numpy as np

if __name__ == '__main__':
    data_dir = '/network/rit/lab/ceashpc/fjie/tmp/ICML2017DVN/FCN/vgg16.npy'

    data = np.load(data_dir, encoding='latin1')
    print(type(data))
    print(data.shape)
    print(data.item().keys())

    data_dict = data.item()

    print(len(data_dict['conv1_1']))
    print(data_dict['conv1_1'][0].shape) # weights
    print(data_dict['conv1_1'][1].shape) # biases