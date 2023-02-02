import torch
import pandas as pd
import numpy as np 
from PIL import Image
from torchvision import transforms

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, istrain):
        self.istrain = istrain
        
        #pandas and readcsv
        df = pd.read_csv('/Users/sreyamuppalla/Desktop/whale-project/data/train/.csv')
        if self.istrain:
            self.images = df.truncate(before = len(df)/5).reset_index(drop = True)
            print(self.images.info())
        else:
            self.images = df.truncate(after = len(df)/5).reset_index(drop = True)
            print(self.images.info())

        self.label_dict = {id: idx for idx, id in enumerate(set(self.images['Id']))}

    def __getitem__(self, index):
        #iloc
        im = Image.open('/Users/sreyamuppalla/Desktop/whale-project/data/train/' + self.images.loc[index ,'Image'])
        im = im.resize((224, 224))
        # im.show()
        converter = transforms.ToTensor()
        inputs = converter(im)
        if inputs.shape[0] == 1:
            inputs = inputs.repeat(3, 1, 1)
        # print(inputs.shape)
        label = self.label_dict[self.images.loc[index, 'Id']]
        return inputs, label

    def __len__(self):
        return len(self.images)