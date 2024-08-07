# 卷积神经网络 

## 1.  padding参数

![image-2024080523420618](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240805234206518.png)

如果为卷积层传入一个padding参数，卷积时会在原数据周围补零

![image-20240805234335577](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240805234335577.png)

## 2. 步长

模型默认步长为1，可以通过指定strike参数来调整

![image-20240805234727479](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240805234727479.png)

调整前，卷积核每次向后移动一个数据，调整后为2个

+ 第一步

![image-20240805234823627](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240805234823627.png)

+ 第二步

  ![image-20240805234852060](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240805234852060.png)



## 3. 下采样

+ 最大池化层（Max Pooling Layer）

  ![image-20240805234520317](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240805234520317.png)

  从原数据的**每一个通道**中，等分4个区域，保留每个区域中**最大**的值

  ![image-20240806000342104](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240806000342104.png)

  ## 4 . 使用CNN训练MNIST数据集

  ![image-20240806000833310](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240806000833310.png)

  1. 先使用`10*5*5`的卷积核接收图像，将`1*28*28`的图像转换为`10*24*24`

  2. 再使用最大池化层, 通道数不变, 宽和高变为原先一半

  3. 再使用`2*5*5`的卷积核, 将数据转换为`20*8*8`

  4. 再使用最大池化层, 通道数不变, 宽和高变为原先一半

  5. 最后使用全连接层, 接收`20*4*4`共320个维度, 输出10个维度,对应0-9

  6. ![image-20240806002010638](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240806002010638.png)

     

### 代码实现

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)
    def forward(self, x):
        # Flatten data from (n, 1, 28, 28) to (n, 784)
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) # flatten
        x = self.fc(x)
        return x
    
# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建模型实例并移动到GPU（如果可用）
model = Net().to(device)

# 使用交叉熵损失，直接接收来自Linear层的输出
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

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
    print('Epoch: %d, Acc on test set: %.4f' % (epoch, correct / total))

if __name__ == '__main__':
    for epoch in range(epochs):
        train(epoch)
        if (epoch + 1) % 10 == 0:
            test(epoch)
```

结果: 

![image-20240806002717769](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240806002717769.png)

