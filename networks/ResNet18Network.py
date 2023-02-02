import torch
import torch.nn as nn

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained = True)
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 25361)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x
