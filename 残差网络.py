
import torch.nn as nn

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
# ResNet
class Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

下面是对代码中每个函数和类的详细解释，包括它们的作用和每个参数的含义。

### 1. `conv3x3` 函数

```python
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
```

#### 作用
- 该函数定义了一个 3x3 的卷积层，并返回一个 `nn.Conv2d` 对象。

#### 参数
- **`in_channels`**: 输入特征图的通道数。例如，对于RGB图像，`in_channels` 通常为3。
- **`out_channels`**: 输出特征图的通道数，即卷积核的数量。
- **`stride`**: 卷积的步幅，决定卷积核在特征图上移动的步长。默认值为1。

#### 其他
- **`kernel_size`**: 卷积核的大小，这里是3x3。
- **`padding`**: 卷积操作的填充大小，这里是1，保证输出特征图的空间尺寸与输入相同。
- **`bias`**: 是否使用偏置项，这里设置为 `False`，因为后面有 Batch Normalization 处理。

---

### 2. `ResidualBlock` 类

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
```

#### 作用
- 该类实现了残差块（Residual Block），是 ResNet 架构的基本组成部分。残差块通过跳过连接（skip connection）解决了深度神经网络中的梯度消失问题。

#### 参数
- **`in_channels`**: 输入特征图的通道数。
- **`out_channels`**: 输出特征图的通道数。
- **`stride`**: 卷积的步幅。
- **`downsample`**: 可选的下采样层，用于调整输入的通道数和特征图的大小，使得残差连接可以进行有效的加法操作。

#### 方法
- **`forward(self, x)`**: 定义了前向传播过程。
  - `residual = x`: 记录输入，以便后续进行残差连接。
  - 依次通过两个卷积层、Batch Normalization 和 ReLU 激活函数处理输入。
  - 如果需要下采样，则更新 `residual`。
  - 最后将 `residual` 加到 `out` 上，并再次通过 ReLU 激活。

---

### 3. `Net` 类（ResNet）

```python
class Net(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

#### 作用
- 该类实现了 ResNet 模型，利用残差块构建深度神经网络。它能够有效地训练更深的网络。

#### 参数
- **`block`**: 残差块的类型，通常是 `ResidualBlock`。
- **`layers`**: 一个列表，包含每层中残差块的数量。
- **`num_classes`**: 最终分类的类别数，默认为10。

#### 方法
- **`make_layer(self, block, out_channels, blocks, stride=1)`**: 
  - 用于构建特定层的多个残差块。
  - **`downsample`**: 如果步幅不为1或者输入和输出通道数不相同，则需要创建一个下采样层。
  - 创建指定数量的残差块，并将它们组合为一个顺序层（`nn.Sequential`）。

- **`forward(self, x)`**: 定义了前向传播过程。
  - 依次通过初始卷积层、Batch Normalization 和 ReLU 激活，然后通过三层残差块。
  - 最后通过平均池化和全连接层进行输出。

