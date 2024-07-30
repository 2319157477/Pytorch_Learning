# 第四课 用Pytorch实现线性回归

## 1.准备数据集

准备好的数据集中的数据应当是矩阵的的形式
![image-20240730221947503](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730221947503.png)



## 2.设计模型（nn.Module)

在构建一个神经网络时，我们需要做的是构建出一张计算图，如下所示

在Pytorch中，一个仿射模型：
$$
\hat{y}=x*w+b
$$
被视作一个线性单元

![image-20240730222155759](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730222155759.png)

其中，权重w应当由y_pred以及x的维度数决定（在这里，矩阵的列数表示维度，即样本特征数；行数表示样本个数）

![image-20240730231306250](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730231306250.png)



## 3.设计损失函数与优化器（Pytorch API）

### 1. 损失函数

![image-20240730232650893](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730232650893.png)

当训练的总轮数不能在最后完成一个完整的batch时，将size_average设置为True

### 2.优化器

![image-20240730232817617](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730232817617.png)

model.parameters()方法会找出模型中所有需要优化的权重，lr（学习率）为固定值



## 4.训练周期

![image-20240730234325161](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730234325161.png)

输出结果：

![image-20240730234355540](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730234355540.png)

## 代码实现

~~~python
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])
epochs = 5000

class LinerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1) #(1, 1)代表输入特征数为1，输出特征数为1

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinerModel()

#损失函数
criterion = torch.nn.MSELoss(reduction='sum')

#优化器
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

#训练
for epoch in range(epochs):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}: loss = {loss.item()}')

    #优化器归零
    optimizer.zero_grad()
    #对loss反向传播
    loss.backward()
    #使用优化器对权重进行一次更新
    optimizer.step()

#输出权重和偏置
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

#测试
x_test = torch.Tensor([[4.0]])
print('y_pred = ', model(x_test).data)
~~~

结果展示：

![image-20240730235233724](C:\Users\23191\AppData\Roaming\Typora\typora-user-images\image-20240730235233724.png)

