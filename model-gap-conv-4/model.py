import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        '''
        Input Block
        In => 28 * 28 * 1
        Out => 26 * 26 * 10
        RF => 3
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        '''
        Convolution Block 2
        In => 26 * 26 * 10
        Out => 24 * 24 * 10
        RF => 5
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        '''
        Transition Block 1
        In => 22 * 22 * 10
        Out => 6 * 6 * 10
        RF => 8
        '''
        self.transitionBlock1 = nn.Sequential(
            nn.MaxPool2d(2,2)
        )

        '''
        Convolution Block 3
        In => 24 * 24 * 10
        Out => 22 * 22 * 10
        RF => 7
        '''
        self.conv3 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1))

        '''
        Convolution Block 4
        In => 6 * 6 * 10
        Out => 4 * 4 * 10
        RF => 12
        '''
        self.conv4 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=3),
        )

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        )

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(40, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transitionBlock1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = x.view(-1, 40)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
