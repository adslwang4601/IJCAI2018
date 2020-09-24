#!/usr/bin/python
# coding=utf-8

import numpy
import random
import scipy.special as special
from tqdm import tqdm
import math
from math import log
import pandas as pd
numpy.random.seed(0)

class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def sample_from_beta(self, alpha, beta, num, imp_upperbound):
        #产生样例数据
        sample = numpy.random.beta(alpha, beta, num)
        I = []
        C = []
        for click_ratio in sample:
            imp = random.random() * imp_upperbound
            #imp = imp_upperbound
            click = imp * click_ratio
            I.append(imp)
            C.append(click)
        return pd.Series(I), pd.Series(C)

    def update(self, tries, success, iter_num, epsilon):
        #更新策略
        for i in tqdm(range(iter_num)):
            new_alpha, new_beta = self.__fixed_point_iteration(tries, success, self.alpha, self.beta)
            if abs(new_alpha-self.alpha)<epsilon and abs(new_beta-self.beta)<epsilon:
                break
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, tries, success, alpha, beta):
        #迭代函数
        sumfenzialpha = 0.0
        sumfenzibeta = 0.0
        sumfenmu = 0.0
        sumfenzialpha = (special.digamma(success+alpha) - special.digamma(alpha)).sum()
        sumfenzibeta = (special.digamma(tries-success+beta) - special.digamma(beta)).sum()
        sumfenmu = (special.digamma(tries+alpha+beta) - special.digamma(alpha+beta)).sum()

        return alpha*(sumfenzialpha/sumfenmu), beta*(sumfenzibeta/sumfenmu)


def test():
    bs = BayesianSmoothing(1, 1)
    I, C = bs.sample(500, 500, 1000, 10000)
    print (I, C)
    bs.update(I, C, 1000, 0.0000000001)
    print (bs.alpha, bs.beta)




if __name__ == '__main__':
    test()