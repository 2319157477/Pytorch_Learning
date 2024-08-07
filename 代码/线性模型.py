import numpy as np;
import matplotlib.pyplot as plt;
from mpl_toolkits.mplot3d import Axes3D;

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [5.0, 8.0, 11.0, 14.0, 17.0]

def forward(x) :
    return x * w + b


def loss(x, y) :
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = [] #存储权重w
b_list = [] #存储偏置b
mse_list = [] #存储权重对应的均方误差
for w in np.arange(0.0, 5.1, 0.1) :
    for b in np.arange(0.0, 3.1, 0.1) :
        print("w = ", w, ", b = ", b)
        loss_sum = 0
        for x_val, y_val in zip(x_data, y_data): 
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            loss_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        mse = loss_sum / len(x_data)
        print("MSE = ", mse)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 确保 w_list, b_list, 和 mse_list 的长度一致
assert len(w_list) == len(b_list) == len(mse_list)

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制三维散点图
scatter = ax.scatter(w_list, b_list, mse_list, c=mse_list, cmap='viridis')

# 添加颜色条
fig.colorbar(scatter, shrink=0.5, aspect=5)

# 设置标签
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE')

plt.show()
