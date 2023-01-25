import torch
import pandas as pd
import numpy as np 
from PIL import Image
from torchvision import transforms

class TrainDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        
        #pandas and readcsv
        df = pd.read_csv('./data/train/.csv')
        self.images = df
        self.label_dict = {id: idx for idx, id in enumerate(set(df['Id']))}

    def __getitem__(self, index):
        #iloc
        im = Image.open("./data/train/" + self.images.loc[index ,'Image'])
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
        return 10000