import torch
import torch.nn as nn
import torch.utils.data as DATA
import torchvision
import os
# import cv2


cnn = torch.load('net.pkl')
print(cnn)


optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()


for epoch in range(EPOCH):
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
                print('Epoch: ', epoch, '| Steps: %03d' % step,  '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)
                break






