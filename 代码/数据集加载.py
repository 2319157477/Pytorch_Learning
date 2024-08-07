import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

epochs = 10000

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1]) #[:,:-1]中，第一个':'表示所有行,':-1'表示排除-1这一行(即y)
        self.y_data = torch.from_numpy(xy[:, [-1]]) #[:, [-1]]中,[-1]表示只需要最后一行,并且存储为矩阵形式

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.len
    
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

#准备数据集    
dataset = DiabetesDataset('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=0)

model = Model() 

#损失函数为BCE（交叉熵损失）
criterion = torch.nn.BCELoss(reduction="mean")
#优化器选择Adam
optimizer = optim.Adam(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            #准备每一个batch的数据
            x, y = data
            #Forward
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if (epoch + 1) % 1000 == 0:
                print(epoch + 1, i, loss.data.item())
            #Backward
            optimizer.zero_grad()
            loss.backward()
            #Update
            optimizer.step()


