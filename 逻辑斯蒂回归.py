import torch
import torch.nn.functional as tf ##引入函数包

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])
epochs = 5000

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        #更改为sigmoid（逻辑斯蒂）函数
        y_pred = tf.sigmoid(self.linear(x))
        return y_pred
    
model = LogisticRegressionModel()

#损失函数变更为BCE（交叉熵损失）
criterion = torch.nn.BCELoss(reduction="sum")
#优化器选择Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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

#测试
x_test = torch.Tensor([[4.0]])
print('y_pred = ', model(x_test).data)
