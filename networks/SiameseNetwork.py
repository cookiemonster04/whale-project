import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

cpu = torch.device('cpu')

class SiameseNetwork(torch.nn.Module):
    """
    Siamese network with cosine distance function
    """
    
    def __init__(self, stage=0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 8, 8, 2, padding=(3,3)) # 3 x 224 x 224 -> 8 x 112 x 112
        self.pool1 = nn.MaxPool2d(4, 2, padding=(1,1)) # 8 x 112 x 112 -> 8 x 56 x 56
        self.conv2 = nn.Conv2d(8, 8, 3, 1, padding='same')
        self.pool2 = nn.MaxPool2d(4, 2, padding=(1,1)) # 8 x 56 x 56 -> 8 x 28 x 28
        self.conv3 = nn.Conv2d(8, 16, 3, 1, padding='same')
        self.pool3 = nn.MaxPool2d(3, 2, padding=(1,1)) # 16 x 14 x 14
        self.conv4 = nn.Conv2d(16, 64, 3, 1, padding='same') # 16 x 14 x 14 -> 64 x 14 x 14
        self.pool4 = nn.MaxPool2d(2, 2) # 64 x 14 x 14 -> 64 x 7 x 7
        self.fc1 = nn.Linear(64*7*7, 500)
        self.fc2 = nn.Linear(500, 500)
        self.relu = nn.ReLU()
        self.loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y))
        if stage > 2:
            self.init_classify_params()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = x+self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = x / torch.linalg.vector_norm(x)
        return x
    
    def init_classify(self, x, y):
        indices = [i for i, yv in enumerate(y) if yv < 5004]
        x = x[indices]; y = y[indices]
        embed = self.forward(x)
        for yv, ev in zip(y, embed):
            self.embed_matrix[yv] += ev / torch.linalg.vector_norm(ev)
            # self.embed_count[yv] += 1
    
    def init_classify_params(self):
        self.embed_matrix = nn.Parameter(torch.zeros([5004, 500], requires_grad=True))
        # self.embed_count = nn.Parameter(torch.zeros([5004, 1], requires_grad=True))
        self.new_thresh = nn.Parameter(torch.tensor(0.5, requires_grad=True))
        self.softmax = nn.Softmax(dim=0)
    
    def norm_embed(self):
        self.embed_matrix /= torch.linalg.vector_norm(self.embed_matrix, dim=0)
        print(self.embed_matrix.size())
        assert(torch.isnan(self.embed_matrix).sum() == 0)
        
    def classify(self, x):
        embed = self.forward(x)
        embed_dist = torch.concat((self.embed_matrix @ embed.T, self.new_thresh.expand(1, embed.size()[0])))
        logits = self.softmax(embed_dist).T
        # torch.set_printoptions(profile="full")
        # print(logits)
        return logits