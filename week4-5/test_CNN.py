import torch
import torch.nn as nn
import torch.utils.data as DATA
import torchvision
import os
import cv2
import math


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


class MNIST:
    def __init__(self):
        self.predict = False
        self.cnn = CNN()
        self.src = cv2.imread("number.jpg")
        self.p1 = ()
        self.p2 = ()

        if torch.cuda.is_available():
            self.cnn.load_state_dict(torch.load('./model/cnn_params_gpu_100_100_0.001.pkl'))
        else:
            self.cnn.load_state_dict(torch.load('./model/cnn_params_gpu_100_100_0.001.pkl', map_location='cpu'))
        self.src = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        if self.src.shape[0] > 1080 or self.src.shape[1] > 1920:
            factor = max(math.ceil(self.src.shape[1] / 1080), math.ceil(self.src.shape[0] / 1920))
            self.src = cv2.resize(self.src, (int(self.src.shape[1] / factor), int(self.src.shape[0] / factor)))

    def show(self):
        if self.p1 and self.p2:
            img = self.src.copy()
            cv2.imshow("MNIST", cv2.rectangle(img, self.p1, self.p2, (0, 0, 0), 2))
        else:
            cv2.imshow("MNIST", self.src)

    def mouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.p1 = (x, y)
            self.p2 = ()

        elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
            self.p2 = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.p1 and self.p2 and self.p1[0] < self.p2[0] and self.p1[1] < self.p2[1]:
                self.predict = True
            else:
                self.p1 = self.p2 = ()

    def main(self):
        print(self.cnn)
        print("Load successfully, waiting...")
        cv2.imshow("MNIST", self.src)
        cv2.setMouseCallback("MNIST", self.mouseCallback)
        while True:
            self.show()
            cv2.waitKey(30)
            if self.predict:
                src = self.src[self.p1[1]:self.p2[1], self.p1[0]:self.p2[0]]
                src = cv2.resize(src, (28, 28))
                thresh, src = cv2.threshold(src, 127, 255, cv2.THRESH_BINARY_INV)

                test_input = torch.from_numpy(src).float()
                test_input = torch.unsqueeze(test_input, 0)
                test_input = torch.unsqueeze(test_input, 0)
                test_output = self.cnn(test_input)
                pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
                print("The number in picture is ", pred_y)
                self.predict = False


if __name__ == '__main__':
    mnist = MNIST()
    mnist.main()

