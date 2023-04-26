import torch
import pandas as pd
import numpy as np 
from PIL import Image
from torchvision import transforms
import time
import random
import os
import constants
from tqdm import tqdm

class StartingDataset(torch.utils.data.Dataset):
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
    def __init__(self):
        #pandas and readcsv
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        self.images = df
        label_set = set(self.images['Id'])
        label_set.discard('new_whale')
        self.label_dict = {id: idx for idx, id in enumerate(sorted(label_set))}
        self.label_dict.update({'new_whale': len(label_set)}) # ensure new_whale is last dimension
        self.nlabel = len(self.label_dict)
        self.label_list = [[] for _ in range(self.nlabel)]
        for idx, id in enumerate(self.images['Id']):
            self.label_list[self.label_dict[id]].append(idx)
        self.time_taken = 0.

    def __getitem__(self, index):
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
        # return (self.nlabel-1)//10
        return self.nlabel-1
    
    def gen_matrix(self, *args):
        pass

class GenDataset(torch.utils.data.Dataset):
    """
    An unsplit dataset used for generating model embeddings for hard triplet mining
    """
    def __init__(self):
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        self.images = df
        self.time_taken = 0.

    def __getitem__(self, index):
        #iloc
        start_time = time.perf_counter()
        im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index ,'Image']))
        im = im.resize((224, 224))
        converter = transforms.ToTensor()
        inputs = converter(im)
        if inputs.shape[0] == 1:
            inputs = inputs.repeat(3, 1, 1)
        end_time = time.perf_counter()
        self.time_taken += end_time-start_time
        return inputs
    
    def __len__(self):
        return len(self.images)

class HardSiameseDataset(torch.utils.data.Dataset):
    def __init__(self, device):
        #pandas and readcsv
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        self.images = df
        self.labels = self.images['Id']
        label_set = set(self.labels) # hashing can be seen as a seeded pseudorng - the same unordered order for the same ids
        label_set.discard('new_whale')
        self.label_dict = {id: idx for idx, id in enumerate(label_set)}
        self.label_dict.update({'new_whale': len(label_set)}) # ensure new_whale is last dimension
        self.nlabel = len(self.label_dict)
        self.label_list = [[] for _ in range(self.nlabel)]
        for idx, id in enumerate(self.images['Id']):
            self.label_list[self.label_dict[id]].append(idx)
        self.time_taken = 0.
        self.embed_matrix = torch.zeros(len(self.images), 500)
        self.device = device
        self.gen_dataset = GenDataset()
        # avail_mem = torch.cuda.get_device_properties(self.device).total_memory / 1024 ** 2 if torch.cuda.is_available() else 0
        # self.gen_batch = 2 ** int(avail_mem / (287.11/500)).bit_length() if torch.cuda.is_available() else 32
        # self.gen_batch = 256 if torch.cuda.is_available() else 32
        self.gen_batch = 256 if torch.cuda.is_available() and self.device == torch.device('cuda') else 32
        
    def __getitem__(self, index):
        start_time = time.perf_counter()
        same_label = index % (self.nlabel-1) # prevent selecting new_whale as anchor
        idxs = random.sample(self.label_list[same_label], k=2) if len(self.label_list[same_label]) > 1 \
            else self.label_list[same_label]*2
        hard_idx = self.query_hard(idxs[0], same_label)
        idxs.append(hard_idx)
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
        inputs.extend(idxs)
        end_time = time.perf_counter()
        self.time_taken += end_time-start_time
        return tuple(inputs)

    def __len__(self):
        # return (self.nlabel-1)//10
        return self.nlabel-1
    
    def gen_matrix(self, model):
        # loader = torch.utils.data.DataLoader(self.gen_dataset, batch_size=self.gen_batch, shuffle=False)
        # pbar = tqdm(enumerate(loader), total=len(loader))
        # model = model.to(self.device)
        # self.embed_matrix = self.embed_matrix.to(self.device)
        # start_idx = 0
        # model.eval()
        # with torch.no_grad():
        #     for it, x in pbar:
        #         x = x.to(self.device)
        #         y = model(x)
        #         self.embed_matrix[start_idx:start_idx+x.size(dim=0)] = y
        #         start_idx += x.size(dim=0)
        #         pbar.set_description(f"iter {it}")
        # model.train()
        try:
            loader = torch.utils.data.DataLoader(self.gen_dataset, batch_size=self.gen_batch, shuffle=False)
            pbar = tqdm(enumerate(loader), total=len(loader))
            model = model.to(self.device)
            self.embed_matrix = self.embed_matrix.to(self.device)
            start_idx = 0
            model.eval()
            with torch.no_grad():
                for it, x in pbar:
                    x = x.to(self.device)
                    y = model(x)
                    self.embed_matrix[start_idx:start_idx+x.size(dim=0)] = y
                    start_idx += x.size(dim=0)
                    pbar.set_description(f"iter {it}")
            model.train()
        except:
            self.gen_batch //= 2
            if self.gen_batch == 0:
                return
            torch.cuda.empty_cache()
            print(torch.cuda.mem_get_info())
            self.gen_matrix(model)
    
    def gen_with_prof(self, model):
        from pytorch_memlab import LineProfiler
        with LineProfiler(self.gen_matrix) as prof:
            self.gen_matrix(model)
        prof.display()

    def query_hard(self, query_idx, query_label):
        with torch.no_grad():
            self.embed_matrix = self.embed_matrix.to(self.device)
            query_vector = self.embed_matrix[query_idx]
            score = (query_vector @ self.embed_matrix.T)
            filter = torch.from_numpy(np.asarray([label != query_label for label in self.labels])).to(self.device)
            score *= filter
            return torch.argmax(score).item()