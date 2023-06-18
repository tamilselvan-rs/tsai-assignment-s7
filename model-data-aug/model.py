import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.transitionBlock1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.conv5 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        self.conv6 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.transitionBlock1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
