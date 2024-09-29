import torch 
import torchvision 
import torchvision.transforms as transforms 


tranforms = transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#下载训练数据
trainset = torchvision.datasets.CIFAR10(root='',tranform = tranforms, train=True, download=True)
# 读取训练数据
trainloder = torch.utils.data.DataLoder(trainset,batch_size = 4,shuffle = True,num_workers = 0)

testset = torchvision.datasets.CIFAR10(root='',tranform = tranforms, test=True, download=True)
testloder = torch.utils.data.DataLoder(testset,batch_size = 4,shuffle = True,num_workers = 0)

#指定类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse','ship', 'truck')

#查看部分图像
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    image = img / 2 + 0.5     # 像素归一化
    npimg = image.numpy() # 将图像原本的tensor形式转换为numpy形式
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # 图像显示
    plt.show()

dataiter = iter(trainloder) # 设置迭代器
images, labels = dataiter.next() #读取第二个batch的数据
imshow(torchvision.utils.make_grid(images)) # 将每一个批次的图片组合在一起
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))#CIFAR-10 数据集,它有10个类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车）,那么 labels[j] 可能是从 0 到 9 的整数之一,用来指示第 j 张图像属于哪个类别



#模型训练+调参
import torch.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, padding=1,stride = 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.fc1 = nn.Linear(18*16*16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        #view将池化后的张量拉伸,-1的意思其实就是未知数的意思,根据其他位置（这里就是18*16*16）来推断这个-1是几
        x = x.view(-1, 18*16*16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#定义损失函数和优化器

import torch.optim as optim

def createlossandoptimizer(net, learning_rate = 0.001):
    loss = nn.CrossEntropyLoss() # 定义损失函数为交叉熵
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9) # 定义优化器为SGD
    return loss, optimizer

#定义随机采样器
from torch.utils.data import RandomSampler
train_sample = RandomSampler(trainset) 
test_sample = RandomSampler(testset) #test_sample = RandomSampler(testset) 
validation_sample = RandomSampler(trainset, num_samples=int(len(trainset) * 0.2), replacement=False) # 随机采样 20% 的训练数据作为验证集
'''val_loader: 创建了一个验证集的数据加载器,批量大小为 64,使用 validation_sample 作为采样器,随机抽取 20% 的训练数据。
test_loader: 创建了一个测试集的数据加载器,批量大小为 4,使用 test_sample 作为采样器,从测试集中随机抽样。'''


def get_train_loader(batch_size):
    # train_loader, 一次性加载了sample中全部的样本数据,每次以batch_size为一组循环
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sample, num_workers=2)   
    return train_loader

val_loader = torch.utils.data.DataLoader(trainset, batch_size=64, sampler=validation_sample, num_workers=2)

test_loader = torch.utils.data.DataLoader(testset, batch_size=4, sampler=test_sample, num_workers=2)
'''train_sample、validation_sample 和 test_sample 都是通过 RandomSampler 创建的,sampler 会根据这些采样器的定义从 trainset 和 testset 中提取数据'''

#迭代,训练模型
import time
def trainNet(net, batchsize, n_epochs, learning_rate):
    print("HYPERPARAMETERS:")  
    print("batch-size=", batchsize)
    print("n_epochs=", n_epochs)
    print("learning_rate=", learning_rate)

    print("batchsize:", batchsize)
    train_loader = get_train_loader(batchsize)
    n_batches = len(train_loader)  # n_batches * batchsize = 20000（样本数目）
    print("n_batches", n_batches)
    loss, optimizer = createlossandoptimizer(net, learning_rate)
 
    training_start_time = time.time() 
    print("training start:")
    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = n_batches 
        print("print_every:", print_every)  
        start_time = time.time()
        total_train_loss = 0
 
        for i, data in enumerate(train_loader, 0):#enumerate(train_loader, 0) 将 train_loader 进行枚举,从索引 0 开始,每次迭代返回一个索引值 i 以及对应的 data
            inputs, labels = data
            # 确保 inputs 需要计算梯度
            inputs, labels = inputs.requires_grad_(), labels

            # 将所有的梯度置零,防止每次 backward 时梯度累加
            optimizer.zero_grad() 
            
            # forward
            outputs = net(inputs)
            # loss
            loss_size = loss(outputs, labels)
            # backward
            loss_size.backward()
            # update weights
            optimizer.step()
            print(loss_size)
            running_loss += loss_size.item()
            print("running_loss:", running_loss)
            total_train_loss += loss_size.item()
            print("total_train_loss:", total_train_loss)
            
            # 在一个 epoch 里,每十组 batchsize 大小的数据输出一次结果
            if (i + 1) % 10 == 0:
                print("epoch{}, {:d} \t traing_loss:{:.2f} took:{:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches),
                    running_loss / 10, time.time() - start_time))
                running_loss = 0.0
                start_time = time.time()
 
        total_val_loss = 0
        
        for inputs, labels in val_loader:
            # 确保 inputs 需要计算梯度
            inputs, labels = inputs.requires_grad_(), labels
 
            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()
        
        # 验证集的平均损失          
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    # 所有的 Epoch 结束,也就是训练结束,计算花费的时间
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
