import torch
import torch.nn as nn

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 8, 8, 2, padding=(3,3)) # 3 x 224 x 224 -> 8 x 112 x 112
        self.pool1 = nn.MaxPool2d(4, 2, padding=(1,1)) # 8 x 112 x 112 -> 8 x 56 x 56
        self.conv2 = nn.Conv2d(8, 8, 3, 1, padding='same')
        self.pool2 = nn.MaxPool2d(4, 2, padding=(1,1)) # 8 x 56 x 56 -> 8 x 28 x 28
        self.conv3 = nn.Conv2d(8, 8, 3, 1, padding='same')
        self.pool3 = nn.MaxPool2d(3, 2, padding=(1,1)) # 8 x 14 x 14
        self.conv4 = nn.Conv2d(8, 32, 3, 1, padding='same') # 8 x 14 x 14 -> 32 x 14 x 14
        self.pool4 = nn.MaxPool2d(2, 2) # 32 x 14 x 14 -> 32 x 7 x 7
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 5005)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x+self.conv2(x)
        x = self.pool2(x)
        x = x+self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
