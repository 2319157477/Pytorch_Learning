import numpy as np
import torch
import torch.optim as optim

xy = np.loadtxt('diabetes.csv.gz', delimiter=',', dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1]) #[:,:-1]中，第一个':'表示所有行,':-1'表示排除-1这一行(即y)
y_data = torch.from_numpy(xy[:, [-1]]) #[:, [-1]]中,[-1]表示只需要最后一行,并且存储为矩阵形式
epochs = 500000

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6) #8维降到6维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.ReLU() #引入激活函数
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.linear1(x)) #进行第一步变换,并且使用非线性激活函数
        x = self.activate(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model() 

#损失函数为BCE（交叉熵损失）
criterion = torch.nn.BCELoss(reduction="sum")
#优化器选择Adam
optimizer = optim.Adam(model.parameters(), lr=0.1)
#学习率下降
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.1)

for epoch in range(epochs):
    #Forward
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if (epoch + 1) % 100 == 0:
        print('Epoch:', epoch + 1, 'Loss:', loss.data.item())
    
    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Update
    optimizer.step()
    scheduler.step() 