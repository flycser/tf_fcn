#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename : run_log
# @Date : 08/21/2018 13:07:52
# @Poject : tf_fcn
# @Author : FEI, hfut_jf@aliyun.com
# @Desc :

import logging
from logging.config import fileConfig

fileConfig(fname='../logging.conf')

if __name__ == '__main__':
    logger = logging.getLogger('fei')
    logging.debug('Fei Jie, hello world!')