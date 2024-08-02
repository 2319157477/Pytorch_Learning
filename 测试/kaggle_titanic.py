import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


batch_size = 64
epochs = 1000

# 定义 Titanic 数据集类
class TitanicDataset_train(Dataset):
    def __init__(self, filepath):
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(filepath)
        self.passenger_ids = df['PassengerId'].values
        # 数据预处理：将字符串类型转换为数值类型（例如，将性别和登船港口转换为数值）
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        # Cabin 列处理：如果为空则输出0，不为空则输出1
        df['Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
        df['Age'].fillna(df['Age'].mean(), inplace=True)  # 填补缺失的年龄数据
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # 填补缺失的登船港口数据
        df['Fare'].fillna(df['Fare'].mean(), inplace=True)  # 填补缺失的票价数据
        # 选择特征列和目标列
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
        target = 'Survived'

        # 将数据转换为 NumPy 数组
        x_data = df[features].values.astype(np.float32)
        y_data = df[[target]].values.astype(np.float32)

        # 将 NumPy 数组转换为 PyTorch 张量
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)
        self.len = len(df)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.passenger_ids[index]

    def __len__(self):
        return self.len
    
class TitanicDataset_test(Dataset):
    def __init__(self, filepath):
        # 使用 pandas 读取 CSV 文件
        df = pd.read_csv(filepath)
        self.passenger_ids = df['PassengerId'].values
        # 数据预处理：将字符串类型转换为数值类型（例如，将性别和登船港口转换为数值）
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
        df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
        # Cabin 列处理：如果为空则输出0，不为空则输出1
        df['Cabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
        df['Age'].fillna(df['Age'].mean(), inplace=True)  # 填补缺失的年龄数据
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # 填补缺失的登船港口数据
        df['Fare'].fillna(df['Fare'].mean(), inplace=True)  # 填补缺失的票价数据
        # 选择特征列和目标列
        features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

        # 将数据转换为 NumPy 数组
        x_data = df[features].values.astype(np.float32)

        # 将 NumPy 数组转换为 PyTorch 张量
        self.x_data = torch.from_numpy(x_data)
        self.len = len(df)

    def __getitem__(self, index):
        return self.x_data[index], self.passenger_ids[index]

    def __len__(self):
        return self.len

# 读入数据集
train_set = TitanicDataset_train('../dataset/titanic/train.csv')
test_set = TitanicDataset_test('../dataset/titanic/test.csv')

# 把数据集加载到Dataloader
train_loader = DataLoader(train_set,
                          shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(test_set,
                         shuffle=False,
                         batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(8, 64)
        self.l2 = torch.nn.Linear(64, 32)
        self.l3 = torch.nn.Linear(32, 16)
        self.l4 = torch.nn.Linear(16, 8)
        self.l5 = torch.nn.Linear(8, 4)
        self.l6 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return torch.sigmoid(self.l6(x))

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)

creterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(epoch):
    model.train()
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets, _ = data
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = creterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        if (batch_idx + 1) % 14 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0

def test():
    model.eval()
    results = []
    with torch.no_grad():
        for data in test_loader:
            inputs, passenger_ids = data
            inputs= inputs.to(device)
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            for pid, pred in zip(passenger_ids, predicted):
                results.append((pid.item(), int(pred.item())))
    
    return results


if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
    results = test()
    df = pd.DataFrame(results, columns=['PassengerId', 'Survived'])
    df.to_csv('测试/predictions.csv', index=False)
