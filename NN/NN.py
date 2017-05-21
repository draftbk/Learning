# coding=utf-8
import math
import random

import numpy as np

# 激活函数
def sigmoid(x):
    # 常用这个
    return 1.0/(1+math.exp(-x))

# 感知机逻辑"与"
def single_and(x1,x2):
    # 这里的参数不一定是(-3,2,2),只要能达到效果都好
    h=sigmoid(-3+2*x1+2*x2)
    if h>0.5:
        return 1
    else:
        return 0

# 感知机逻辑"或"
def single_or(x1,x2):
    # 这里的参数不一定是(-1,2,2),只要能达到效果都好
    h=sigmoid(-1+2*x1+2*x2)
    if h>0.5:
        return 1
    else:
        return 0

# 感知机逻辑"非"
def single_not(x1,x2):
    # 这里的参数不一定是(5,-6,0),只要能达到效果都好 这是是算 x1的非,所以 x2 的权重设为 0
    h=sigmoid(5+-6*x1+0*x2)
    if h>0.5:
        return 1
    else:
        return 0

# 前三个感知机的学习,训练出 与(1) 或(2) 非(3) 对应的 w 值的方法
def learn_perceptron(type):
    # 学习率
    n=0.15
    # 初始权重(也可以随机,为了方便我就取了 1,但不都取1比较好) 代表 w1 ,w2
    w=np.array([3.0,2.0])
    # 偏置为-1,若算 非 的 w 值 bia 要取 +1
    bias=-1
    if type==1:
        # 测试集,[1,1,0]代表 x1=1,x2=1,正确结果是 0 ,训练 与 的测试集
        trainsets = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    if type==2:
        # 训练 或 的测试集
        trainsets = np.array([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]])
    if type==3:
        # 训练 非 的测试集
        trainsets = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 1], [0, 0, 1]])
        bias=1
    err=1
    while err!=0:
        err=0
        for i in range(20):
            trainset=trainsets[random.randint(0, 3)]
            h = sigmoid( w[0] * trainset[0] + w[1] * trainset[1]+bias)
            if h > 0.5:
                deltaW = n * (trainset[2] - 1) * trainset[0:2]
                w = w + deltaW
                err=err+max(abs(deltaW))

            else:
                deltaW = n * (trainset[2] - 0) * trainset[0:1]
                err = err + max(abs(deltaW))
                w = w + deltaW
    return w

# 线性不可分的异或 函数-->需要隐藏层
def single_nor(x1,x2):
    # 这里就是两层了,可能看不清楚,最好看图理解,0.499999是为了解决 0.5 的情况
    h=sigmoid(1*int(sigmoid(1*x1+-1*x2)+0.499999)+int(1*sigmoid(-1*x1+1*x2)+0.499999))
    if h>0.5:
        return 1
    else:
        return 0

# 这三个只要输入层和输出层,并且根据调参的不同能实现不同结果,但是没法实现 --> 异或
print single_and(0,0)
print single_or(1,0)
print single_not(0,0)

# 前三个感知机的学习,训练出 与(1) 或(2) 非(3) 对应的 w 值的方法
print learn_perceptron(1)


# 线性不可分的异或-->需要隐藏层
print single_nor(0,1)

