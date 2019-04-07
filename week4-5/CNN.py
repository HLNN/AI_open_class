import torch
import torch.nn as nn
import torch.utils.data as DATA
import torchvision
import os
import time


EPOCH = 100
BATCH_SIZE = 100
LR = 0.001
DOWNLOAD_MNIST = False

if not (os.path.exists('./mnist/') and os.listdir('./mnist/')):
    if not os.path.exists('./mnist/'):
        os.makedirs('./mnist/')
    DOWNLOAD_MNIST = True


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

