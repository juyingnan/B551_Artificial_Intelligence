#!/bin/bash/env python
# -*- coding: utf-8
__author__ = 'ZhangJin'

import numpy as np


class BPNN:
    def __init__(self, input_array, output_array, layer_node_count_list, w_step_size=0.02, b_step_size=0.02):
        # layers记录每一层的输出,网络是从第0层开始算的
        self.layers = []
        self.input_array = input_array
        self.output_array = output_array
        self.layer_node_count_list = layer_node_count_list
        self.w_step_size = w_step_size
        self.b_step_size = b_step_size
        # layer_count是网络的所有层数减1
        self.layer_count = len(self.layer_node_count_list) - 1
        self.w_dic = {}
        # w_dic是一个dictionary，记录每个层的权重（除了输入层，它没有权重和bias)
        self.b_dic = {}
        # 初始化所有的w和b，
        for i in range(1, len(self.layer_node_count_list)):
            self.w_dic[i] = 2 * (np.random.random((layer_node_count_list[i], layer_node_count_list[i - 1])) - 0.5)
            self.b_dic[i] = 2 * (np.random.random((1, layer_node_count_list[i])) - 0.5)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        else:
            return 1 / (1 + np.exp(-x))

    def forward(self):
        self.layers = []
        self.layers.append(self.input_array)
        for i in range(self.layer_count):
            z = np.dot(self.layers[i], self.w_dic[i + 1].T) + self.b_dic[i + 1]
            a = self.sigmoid(z)
            self.layers.append(a)

    def backward(self):
        delta_list = []
        theta_output = (self.output_array - self.layers[-1]) * self.sigmoid(self.layers[-1], derivative=True)
        #  (self.layers[-1]*(1-self.layers[-1]))
        delta_list.append(theta_output)
        for i in range(self.layer_count - 1, 0, -1):
            theta = np.dot(delta_list[-1], self.w_dic[i + 1]) * self.sigmoid(self.layers[i], derivative=True)
            # (self.layers[i]*(1-self.layers[i]))
            delta_list.append(theta)

        delta_list.reverse()
        w_change_dic = {}
        b_change_dic = {}
        N = len(self.input_array)
        for i in range(len(delta_list)):
            w_change_dic[i + 1] = np.dot(delta_list[i].T, self.layers[i]) / float(N) * self.w_step_size
            b_change_dic[i + 1] = np.sum(delta_list[i], 0) / float(N) * self.b_step_size

        for i in w_change_dic.keys():
            self.w_dic[i] += w_change_dic[i]
            self.b_dic[i] += b_change_dic[i]


if __name__ == '__main__':
    from sklearn.datasets import make_moons

    x, y = make_moons(250, noise=0.25)
    N = 250
    yy = np.reshape(y, [N, 1])
    bpnn = BPNN(x, yy, [2, 3, 4, 1], w_step_size=0.5, b_step_size=0.5)

    for i in range(10000):
        bpnn.forward()
        bpnn.backward()
    print(bpnn.layers[-1])
    cnt = 0
    # 计算准确率
    for i in range(N):
        if np.abs(yy[i] - bpnn.layers[-1][i]) <= 0.5:
            cnt += 1
    print(cnt / float(N))
