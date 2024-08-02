# 第八课 多分类问题（Softmax Classifier）

## 1. 引入

![image-20240801224003831](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240801224003831.png)

对于MINST数据集，我们需要将数字分为10类

![双分类示意](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240801224052166.png)

如果使用Sigmoid对每一个预测进行双分类，那么各个预测的概率是独立的，如$\hat{y}_1$可能为0.8，但$\hat{y}_2$可能为0.9, 这与我们希望的结果不符合.所以我们引入Softmax算子,使得每一个输出的概率都大于零,且和等于1.
$$
P(y=i)= \frac{{e}^{z_i}}{\sum_{j=0}^{K-1}{e}^{z_j}}, i\in\{0,\cdots,K-1\}
$$
## 2. Softmax函数

![Softmax](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240801231220955.png)

在模型中, Softmax层接收来自线性层的输出

![NLLLoss](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240801233506862.png)

在这里

+ **独热编码(one-hot)**将每个类别表示为长度为 N 的二进制向量，只有一个位置为 1，其余位置为 0。

+ **NLLLoss(负对数似然损失)**接收来自Softmax的对数值, 经乘独热编码后, 最后实际的表达式为$-\log\hat{Y}$

  

使用numpy实现:

![numpy](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240801234815215.png)

使用torch实现:

![torch](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240801234923502.png)

其中, `torch.nn.CrossEntropyLoss()`同时包含`Softmax()`函数和`NLLLoss()`函数, 所以在网络的最后一层,无需使用非线性变换, 直接将线性模型的输出输入给`torch.nn.CrossEntropyLoss()`

## 3. 在MNIST上的模型实现

1. **读入数据并转化格式**

![image-20240802023420002](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802023420002.png)

![image-20240802023432435](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802023432435.png)

![image-20240802023814179](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802023814179.png)

2. **设计模型**

   ![image-20240802023858549](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802023858549.png)

3. **设计损失函数和优化器**

   ![image-20240802132400285](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802132400285.png)

4. **设计训练周期**

   ![image-20240802132549690](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802132549690.png)

   ![image-20240802132652660](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240802132652660.png)

## 4. 代码实现

~~~python
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
        super().__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

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
~~~







