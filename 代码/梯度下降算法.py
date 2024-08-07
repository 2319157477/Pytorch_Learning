import numpy as np;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;

w = 1.0 #初始权重
b = 2 #初始偏置
lr = 0.01 #学习率
x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [5.0, 8.0, 11.0, 14.0, 17.0]

#表示模型
def forward(x) :
    return x * w + b

#损失函数
def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

#表示梯度
def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w + b - y)
    return grad / len(xs)

#下降过程
print("训练前预测：", 4, forward(4))

for epoch in range(10000):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= lr * grad_val
    print("Epoch: ", epoch + 1, "w = ", w, "loss = ", cost_val)

print("训练后预测：", 4, forward(4))
