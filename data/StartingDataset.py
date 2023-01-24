import torch
import pandas as pd
import numpy as np 
from PIL import Image

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        
        #pandas and readcsv
        df = pd.read_csv('/Users/jef/Desktop/winter 23 projects/whale-project/humpback-whale-identification/train.csv')
        self.images = df

    def __getitem__(self, index):
        #iloc
        im = Image.open("/Users/jef/Desktop/winter 23 projects/whale-project/humpback-whale-identification/train/" + self.images.loc[index ,'image_id'])
        im = im.resize((224, 224))
       
        im.show()
        inputs = torch.zeros([3, 224, 224])
        label = 0

        return inputs, label

    def __len__(self):
        return 10000
