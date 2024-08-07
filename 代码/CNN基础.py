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