import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

batch_size = 64
epochs = 100

# 将图像转为适合PyTorch处理的C(channel)=1, W(width)=28, H(height)=28的格式, 像素值由{0,...,255}映射到[0, 1]
transform = transforms.Compose([
    transforms.ToTensor(), # 将PIL图像转换成Tensor
    transforms.Normalize((0.1307, ), (0.3081, )) # 通过整个MNIST数据集得出的均值和标准差
])

# 从MNIST加载训练集并且进行格式转换
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)

# 从MNIST加载测试集并且进行格式转换
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)

# 将两个数据集加载到Dataloader中
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_loader = DataLoader(test_dataset,
                          shuffle=False,
                          batch_size=batch_size)

class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(in_channels,16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)
    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)
        self.mp = nn.MaxPool2d(2)
        self.fc_1 = nn.Linear(1408, 20)
        self.fc_2 = nn.Linear(20, 10)
    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)
        return x

    
# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例并移动到GPU（如果可用）
model = Net().to(device)

# 使用交叉熵损失，直接接收来自Linear层的输出
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data
        # 将数据移动到GPU（如果可用）
        inputs, targets = inputs.to(device), targets.to(device)

        # forward + backward + update
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (batch_idx + 1) % 300 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0

def test(epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 遍历测试数据加载器 test_loader，每次获取一个批次的测试数据 data
        for data in test_loader:
            # 将数据拆分成输入图像 images 和对应的标签 labels
            images, labels = data
            # 将数据移动到GPU（如果可用）
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # 使用 torch.max 找到输出张量 outputs 中每行最大值的索引，这些索引就是模型预测的类别标签 predicted
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            # 比较模型预测的标签 predicted 和真实标签 labels，统计预测正确的样本数并累加到 correct 变量中
            correct += (predicted == labels).sum().item()
    print('Epoch: %d, Acc on test set: %.4f' % (epoch + 1, correct / total))

if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        if (epoch + 1) % 10 == 0:
            test(epoch)