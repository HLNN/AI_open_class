import torch
import torch.nn as nn
import torch.utils.data as DATA
import torchvision
import os
import cv2


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
cnn.load_state_dict(torch.load('./model/cnn_params_gpu_100_100_0.001.pkl'))
print(cnn)


src = cv2.imread("5.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src = cv2.resize(src, (28, 28))
thresh, src = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("", src)

test_input = torch.from_numpy(src).float()
test_input = torch.unsqueeze(test_input, 0)
test_input = torch.unsqueeze(test_input, 0)
test_output = cnn(test_input)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print("The number in picture is ", pred_y)

