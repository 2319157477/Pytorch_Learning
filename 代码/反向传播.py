import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0, 4.0, 5.0]
y_data = [8.0, 19.0, 36.0, 59.0, 88.0]

#学习率
lr = 0.0005
epochs = 10000

#权重1
w_1 = torch.tensor([1.0], requires_grad = True)
#权重2
w_2 = torch.tensor([1.0], requires_grad = True)
#偏置
b = torch.tensor([1.0], requires_grad = True)

#此处"*"代表Tensor之间的数乘
def forward(x):
    return x * x * w_1 + x * w_2 + b

#这个函数实际构建了一个计算图
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("训练前的预测：", 4, forward(6).item())
# 用于存储每个 epoch 的损失值
loss_values = []

for epoch in range(epochs):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward() #计算图在进行反向传播后即被释放
        #print('\tgrad:', x, y, w.grad.item())
        with torch.no_grad():
            w_1 -= lr * w_1.grad
            w_2 -= lr * w_2.grad
            b -= lr * b.grad
        w_1.grad.zero_() #将梯度清零
        w_2.grad.zero_()
        b.grad.zero_() 

    loss_values.append(l.item())
    print("Epoch:", epoch + 1, "loss:", l.item())

print("训练后的预测：", 4, forward(6).item())

plt.plot(range(1, epochs + 1), loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss By Epoch')
plt.show()

