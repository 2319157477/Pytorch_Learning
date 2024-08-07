import numpy as np;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;
import random as rd

w = 1.0 #初始权重
b = 2 #初始偏置
lr = 0.01 #学习率
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [5.0, 8.0, 11.0, 14.0, 17.0]

#表示模型
def forward(x) :
    return x * w + b

#损失函数
def loss(x, y):
    y_pred = forward(x)
    loss = (y_pred - y) ** 2
    return loss
    

#表示梯度
def gradient(x, y):
    grad = 2 * x * (x * w + b - y)
    return grad

#下降过程
print("训练前预测：", 4, forward(4))

for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        r = rd.randint(0, 2) #体现随机过程（实际应该随机选择样本）
        if (r == 1): w -= lr * grad
        l = loss(x, y)
    print("Epoch: ", epoch + 1, "w = ", w, "loss = ", l)

print("训练后预测：", 4, forward(4))
