import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, 1, 2) # in, out, kernel, stride, padding 
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, 1, 2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 5, 1, 2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 5, 1, 2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, 1, 2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, 2) # Channel x width x height
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        # Stage 1
        x = self.prelu1_1(self.conv1_1(x)) # 28
        x = self.prelu1_2(self.conv1_2(x)) # 28
        x = F.max_pool2d(x, 2, 2, 0) # 14

        # Stage 2
        x = self.prelu2_1(self.conv2_1(x)) # 14
        x = self.prelu2_2(self.conv2_2(x)) # 14
        x = F.max_pool2d(x, 2, 2, 0) # 7

        # Stage 3
        x = self.prelu3_1(self.conv3_1(x)) # 7
        x = self.prelu3_2(self.conv3_2(x)) # 7
        x = F.max_pool2d(x, 2, 2, 0) # 3

        # Flatten
        # x = torch.flatten(x, 1)
        # x = x.view(x.size(0), -1)
        x = x.view(-1, 128 * 3 * 3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)
        
        return x, y