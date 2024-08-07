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

