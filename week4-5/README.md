# AI开放实验课第四五周总结

-------------------


## 目标

 - 使用卷积神经网络(CNN)完成MNIST手写数字识别
 - 完成模型的保存读取和测试


-----------------


### 安装`pytorch`

作为一个python模块，`pytorch`可以直接用`pip`安装。但针对不同的安装平台，具体的安装包会有不同，所以安装前最好在[`pytorch`官网](https://pytorch.org/get-started/locally/)查一下。
如果下载的网络不好，可以先下载之后再使用`pip`本地安装。


---------------


### CNN卷积神经网络

**定义模型**

 在`CNN.py`中定义力最基本的基于CNN卷积神经网络的手写数字识别。
 
 
首先定义力一些超参数，包括训练的轮数，训练的batch大小，学习率等。

```python
EPOCH = 100
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False

if not (os.path.exists('./mnist/') and os.listdir('./mnist/')):
    if not os.path.exists('./mnist/'):
        os.makedirs('./mnist/')
    DOWNLOAD_MNIST = True
```


**读取数据**

使用`pytorchvision`模块来准备训练和测试的数据集。

```python
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False,
)

train_loader = DATA.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DATA.DataLoader(dataset=train_data, batch_size=1000, shuffle=True)
```


**定义模型**

定义自己的模型结构，我使用的最简单的卷积神经网络。使用了两个卷积模块用来提取图像特征，然后使用一层全连接网络进行分类。卷积模块包括一个二维的卷积层，其输出经过`ReLU`激励函数之后传给一个二维池化层。

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                16, 32, 5, 1, 2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
```


**训练模型**

定义好模型之后就开始训练模型。
首先实例化一个网络，定义优化器`optimizer`和损失函数`loss_func`，然后以`BATCH_SIZE`大小作为一步训练`EPOCH`轮。在每一轮中，每五十步测试一下模型的准确率。

```python
cnn = CNN()
print(cnn)


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    start_time = time.time()
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            for s, (b_x, b_y) in enumerate(test_loader):
                test_output = cnn(b_x)
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(b_y.size(0))
                print('Epoch: ', epoch, '| Steps: %03d' % step,  '| train loss: %.6f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
                break
    end_time = time.time()
    duration = end_time - start_time
    print('Epoch %d' % epoch, ' used %.2f s' % duration)
```


------------------


### 保存载入模型


模型的保存和读取相对简单，但也有一些坑。

**保存模型**

保存模型的时候可以选择保存整个模型或者只保存网络参数。

```python
torch.save(cnn, './model/cnn_gpu_{}_{}_{}.pkl'.format(EPOCH, BATCH_SIZE, LR))
torch.save(cnn.state_dict(), './model/cnn_params_gpu_{}_{}_{}.pkl'.format(EPOCH, BATCH_SIZE, LR))
```


**载入模型**

模型载入和保存一样，也有载入整个网络或者只载入网络参数两种。根据pytorch官网的[文档](https://pytorch.org/docs/master/notes/serialization.html),推荐重新定义一遍模型之后再仅载入参数。
最后的`map_location`是指定模型路径，这样就可以在一台只有CPU的机器上载入使用GPU训练的模型。

```python
self.cnn = CNN()
self.cnn.load_state_dict(torch.load('./model/cnn_params_gpu_100_100_0.001.pkl', map_location='cpu'))
```


## 测试网络

最后我还写了一个`test——CNN.py`用来可视化的测试代码，在显示的测试图片上，用鼠标框选一张图片，就能输出识别的结果。识别完之后在继续框选其他的数字，能继续识别。


![test](https://github.com/HLNN/AI_open_class/blob/master/week4-5/pic/test.png)

