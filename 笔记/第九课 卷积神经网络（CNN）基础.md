# 第9课 卷积神经网络（CNN）

## 1. 引入

在上一课对于MNIST的多分类训练中，我们使用全连接网络，它会把`1*28*28`的图像信息转换成一维的张量，从而导致图像的空间信息被破坏。而卷积神经网络能够将图像以原始的信息保存，不会破坏图像的空间信息

## 2. 网络结构

![image-20240803132232230](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803132232230.png)

先使用卷积层进行特征提取，再使用全连接层进行分类

![image-20240803153208461](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803153208461.png)

卷积的过程可以看作：用一个指定大小的卷积核与原信息中相同大小的区域中的元素做**数乘**，并将结果输出

![image-20240803153353248](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803153353248.png)

这便是对单通道的一次**卷积**

![image-20240803153636443](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803153636443.png)

对于输入的每一个通道，都要分别设置一个卷积核

![](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803153910792.png)

随后将矩阵求和

![image-20240803154213859](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803154213859.png)

由此变换， 我们将一张**3通道的`5*5`**的图像经过**3通道的`3*3`**的卷积核转化成了**单通道`3*3`**的输出

![](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803154717768.png)

所以,如果我们对一张图像同时应用**m个**不同的卷积核, 最后就能得到**m通道**的输出

![image-20240803155445770](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803155445770.png)

总结规律, 如果我们想要将$n\times width_{in}\times height_{in}$的输入经过**卷积**得到$m\times width_{out} \times height_{out}$的输出, 我们就需要一个四维张量

$m \times n \times kernel\_size_{width} \times kernel\_size_{height}$来作为**卷积核**

## 3. 代码实现

~~~python
import torch
in_channels, out_channels = 5, 10 # 指定输入通道和输出通道
width, height = 100, 100 # 指定输入图像的宽和高
kernel_size = 3 # 输入单个数字n默认为nxn的卷积核，也可输入元组构建长方形卷积核
batch_size = 1

# 使用torch.randn()函数生成一个通道、宽、高分别为指定值的随机张量
input = torch.randn(batch_size, in_channels, width, height)

# 为2维卷积层指定三个参数：输入的通道数、输出的通道数、卷积核的大小
# 卷积层的维度取决于需要卷积的数据类型，例如音频就用一维，图像用二维，视频用三维
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
~~~

输出结果：

![image-20240803234932169](https://raw.githubusercontent.com/2319157477/img_bed/main/img/image-20240803234932169.png)

