import torch.nn as nn
import torch.nn.functional as F

'''
Model Params
------------
Best Train Accuracy: 99.05%
Best Test Accuracy: 99.43%
# of Parameters: 17,332
RF Out: 26
Batch Size: 128
LR: 0.01

Target
------
- 99.4% Accuracy
- Improve Train Accuracy further to 99%
- No limit on number of parameters
- RF >= 20

Insights
--------
- Reducing batch size and adjust LR has significantly helped in improving train accuracy
- Model isn't overfitting
- Number of parameters are too high for the assignment target
- Model is beautiful primarily because the train accuracy improvements are transferred to test accuracy
'''

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()

        '''     
        Input Block
        In => 28 * 28 * 1
        Out => 26 * 26 * 32
        RF => 3
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        '''
        Considering the pixels are continuous taking an average at RF3 should help cutdown unnecessary whites 
        Transition Block 1
        In => 26 * 26 * 32
        Out => 13 * 13 * 32
        Pooling Layer => Average Pooling
        Stride In => 2
        Jin => 1
        Jout => 2
        RF => 4
        '''
        self.transitionBlock1 = nn.AvgPool2d(2, 2)

        '''
        Conv 2
        In => 13 * 13 * 32
        Out => 11 * 11 * 32
        Stride In => 1
        Jin => 2
        Jout => 2
        RF => 8 
        '''
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
     
        '''
        Transition Block 2
        RF 8 => Possibly edges are detected
        Transition Block 2
        In => 11 * 11 * 32
        Out => 6 * 6 * 32
        Stride In => 2
        Jin => 2
        Jout => 4
        RF => 10 
        '''
        self.transitionBlock2 = nn.MaxPool2d(2,2,padding=1)
        
        
        '''
        Conv Block 3
        Squeeze after an expansion
        In => 6 * 6 * 32
        Out => 4 * 4 * 16
        Stride In => 1
        Jin => 4
        Jout => 4
        RF => 18 
        '''
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        '''
        Conv Block 4 - Output Layer
        In => 4 * 4 * 16
        Out => 2 * 2 * 16
        Strde In => 1
        Jin => 4
        Jout => 4
        RF => 26
        '''
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.transitionBlock1(x)
        x = self.conv2(x)
        x = self.transitionBlock2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 64)
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)
