import torch
import pandas as pd
import numpy as np 
from PIL import Image
from torchvision import transforms
import time
import random
import os
import constants

class StartingDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, istrain):
        self.istrain = istrain
        
        #pandas and readcsv
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        if self.istrain:
            self.images = df.truncate(before = len(df)/5).reset_index(drop = True)
        else:
            self.images = df.truncate(after = len(df)/5).reset_index(drop = True)

        label_set = set(df['Id']) # hashing can be seen as a seeded pseudorng - the same unordered order for the same ids
        label_set.discard('new_whale')
        self.label_dict = {id: idx for idx, id in enumerate(label_set)}
        self.label_dict.update({'new_whale': len(label_set)}) # ensure new_whale is last dimension
        
        self.transformations = transforms.RandomApply([transforms.Grayscale(3), transforms.RandomCrop((224,224)),
                           transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                           transforms.GaussianBlur(5)], p=0.3)
        self.time_taken = 0.

    def __getitem__(self, index):
        #iloc
        start_time = time.perf_counter()
        im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index ,'Image']))
        im = im.resize((224, 224))
        # im.show()
        converter = transforms.ToTensor()
        inputs = converter(im)
        if inputs.shape[0] == 1:
            inputs = inputs.repeat(3, 1, 1)
        # print(inputs.shape)
        label = self.label_dict[self.images.loc[index, 'Id']]
        inputs = self.transform(inputs)
        end_time = time.perf_counter()
        self.time_taken += end_time-start_time
        return inputs, label

    def transform(self, image):
        # return image
        return self.transformations(image)
    def __len__(self):
        return len(self.images)
    
class SiameseDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self):
        #pandas and readcsv
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        self.images = df
        label_set = set(self.images['Id']) # hashing can be seen as a seeded pseudorng - the same unordered order for the same ids
        label_set.discard('new_whale')
        self.label_dict = {id: idx for idx, id in enumerate(label_set)}
        self.label_dict.update({'new_whale': len(label_set)}) # ensure new_whale is last dimension
        self.nlabel = len(self.label_dict)
        self.label_list = [[] for _ in range(self.nlabel)]
        for idx, id in enumerate(self.images['Id']):
            self.label_list[self.label_dict[id]].append(idx)
        self.time_taken = 0.

    def __getitem__(self, index):
        #iloc
        start_time = time.perf_counter()
        same_label = index % (self.nlabel-1) # prevent selecting new_whale as anchor
        diff_label = random.randint(0, self.nlabel-2) # equal sampling for each whale type
        diff_label = diff_label if diff_label != same_label else self.nlabel-1 # okay to select new_whale
        idxs = random.sample(self.label_list[same_label], k=2) if len(self.label_list[same_label]) > 1 \
            else self.label_list[same_label]*2
        idxs.extend(random.sample(self.label_list[diff_label], k=1))
        inputs = []
        for index in idxs:
            im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image']))
            im = im.resize((224, 224))
            # im.show()
            converter = transforms.ToTensor()
            input = converter(im)
            if input.shape[0] == 1:
                input = input.repeat(3, 1, 1)
            inputs.append(input)
        end_time = time.perf_counter()
        self.time_taken += end_time-start_time
        return tuple(inputs)

    def __len__(self):
        return self.nlabel*5