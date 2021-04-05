#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/4/5 17:44
# @Author  : JJkinging
# @File    : PMF_SGD.py
# -------------------------FUNCTION---------------------------#
from pylab import *
import numpy as np
import random
import math


def SGD(train, test, N, M, learning_rate, K, lambda_1, lambda_2, Step):
    # train: train data
    # test: test data
    # N:the number of user
    # M:the number of item
    # learning_rate: the learning rate
    # K: the number of latent factor
    # lambda_1,lambda_2: regularization parameters
    # Step: the max iteration
    # 分别用正态分布初始化U、V矩阵
    U = np.random.normal(0, 0.1, (N, K))
    V = np.random.normal(0, 0.1, (M, K))
    rmse = []
    loss = []
    for ste in range(Step):
        los = 0.0
        for data in train:
            u = data[0]  # user id，代表某一用户
            i = data[1]  # item id，代表某一项目
            r = data[2]  # rating score # 该项评分

            # error = rating score - predict score, 其中, predict = U * V^T
            e = r - np.dot(U[u], V[i].T)
            # 对U、V矩阵进行梯度更新
            U[u] = U[u] + learning_rate * (e * V[i] - lambda_1 * U[u])
            V[i] = V[i] + learning_rate * (e * U[u] - lambda_2 * V[i])

            # 计算损失
            los = los + 0.5 * (e ** 2 + lambda_1 * np.square(U[u]).sum() + lambda_2 * np.square(V[i]).sum())
        loss.append(los)
        # 对这一轮训练进行测试
        rms = RMSE(U, V, test)
        rmse.append(rms)

        if ste % 10 == 0:
            print(' step:%d | loss:%.4f | rmse:%.4f' % (ste, los, rms))

    return loss, rmse, U, V

# 根均方误差的计算
def RMSE(U, V, test):
    count = len(test)
    sum_rmse = 0.0
    for t in test:
        u = t[0]
        i = t[1]
        r = t[2]
        pr = np.dot(U[u], V[i].T)
        sum_rmse += np.square(r - pr)
    rmse = np.sqrt(sum_rmse / count)
    return rmse

# 加载数据
def Load_data(filedir, ratio):
    '''
    :param filedir: u.data 的路径
    :param ratio: 划分训练集和测试集的比例
    :return:
    '''
    user_set = {}
    item_set = {}
    N = 0;  # the number of user
    M = 0;  # the number of item
    u_idx = 0
    i_idx = 0
    data = []
    f = open(filedir)
    for line in f.readlines():
        u, i, r, t = line.split()
        if int(u) not in user_set:
            user_set[int(u)] = u_idx
            u_idx += 1
        if int(i) not in item_set:
            item_set[int(i)] = i_idx
            i_idx += 1
        data.append([user_set[int(u)], item_set[int(i)], int(r)])
    f.close()
    N = u_idx;
    M = i_idx;

    np.random.shuffle(data)
    train = data[0:int(len(data) * ratio)]
    test = data[int(len(data) * ratio):]
    return N, M, train, test

# 画图
def Figure(loss, rmse):
    fig1 = plt.figure('LOSS')
    x = range(len(loss))
    plot(x, loss, color='g', linewidth=3)
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('Loss.png')

    fig2 = plt.figure('RMSE')
    x = range(len(rmse))
    plot(x, rmse, color='r', linewidth=3)
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.savefig('RMSE.png')
    show()


# ----------------------------SELF TEST----------------------------#

def main():
    dir_data = "./dataset/ml-100k/u.data"
    ratio = 0.8
    N, M, train, test = Load_data(dir_data, ratio)

    learning_rate = 0.005
    K = 10
    lambda_1 = 0.1
    lambda_2 = 0.1
    Step = 100
    loss, rmse, U, V = SGD(train, test, N, M, learning_rate, K, lambda_1, lambda_2, Step)
    Figure(loss, rmse)


if __name__ == '__main__':
    main()