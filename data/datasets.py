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
from torchvision.models.convnext import ConvNeXt_Tiny_Weights
import constants
from queue import PriorityQueue
import logging

class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, istrain):
        self.istrain = istrain
        self.converter = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
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
        # with Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image'])) as im:
        #     inputs = self.converter(im.convert('RGB'))
        im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image']))
        inputs = self.converter(im.convert("RGB"))
        im.close()
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
        self.converter = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
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
            # with Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image'])) as im:
            #     input = self.converter(im.convert('RGB'))
            im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image']))
            input = self.converter(im.convert("RGB"))
            im.close()
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
        self.converter = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        self.images = df
        self.time_taken = 0.

    def __getitem__(self, index):
        #iloc
        im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image']))
        inputs = self.converter(im.convert("RGB"))
        im.close()
        return inputs
    
    def __len__(self):
        return len(self.images)

class HardSiameseDataset(torch.utils.data.Dataset):
    def __init__(self, device):
        #pandas and readcsv
        self.converter = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        self.images = df
        label_set = set(self.images['Id'])
        label_set.discard('new_whale')
        label_set = sorted(label_set) # sorted to get a deterministic result
        self.label_dict = {id: idx for idx, id in enumerate(label_set)}
        self.label_dict.update({'new_whale': len(label_set)}) # ensure new_whale is last dimension
        self.nlabel = len(self.label_dict)
        self.labels = [self.label_dict[id] for id in self.images['Id']]
        self.label_list = [[] for _ in range(self.nlabel)]
        for idx, id in enumerate(self.labels):
            self.label_list[id].append(idx)
        self.time_taken = 0.
        self.embed_matrix = torch.zeros(len(self.images), 500)
        self.device = device
        self.gen_dataset = GenDataset()
        self.gen_batch = 128 if constants.ENV != "local" else (128 if torch.cuda.is_available() and self.device == torch.device('cuda') else 32)
        self.loader = torch.utils.data.DataLoader(self.gen_dataset, batch_size=self.gen_batch, shuffle=False)
        self.losses = []
        self.best_loss = 0
        
    def __getitem__(self, index):
        start_time = time.perf_counter()
        same_label = index % (self.nlabel-1) # prevent selecting new_whale as anchor
        idxs = random.sample(self.label_list[same_label], k=1)
        hard_idx = self.query_hard(idxs[0], same_label)
        idxs.extend(hard_idx)
        inputs = []
        for index in idxs:
            im = Image.open(os.path.join(constants.TRAIN_PATH, self.images.loc[index, 'Image']))
            input = self.converter(im.convert("RGB"))
            im.close()
            inputs.append(input)
        inputs.extend(idxs)
        end_time = time.perf_counter()
        self.time_taken += end_time-start_time
        return tuple(inputs)

    def __len__(self):
        return self.nlabel-1
    
    def gen_matrix(self, model):
        pbar = tqdm(enumerate(self.loader), total=len(self.loader))
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
    
    def query_hard(self, query_idx, query_label):
        with torch.no_grad():
            self.embed_matrix = self.embed_matrix.to(self.device)
            query_vector = self.embed_matrix[query_idx]
            score = (query_vector @ self.embed_matrix.T)
            score += torch.from_numpy(np.asarray([1000 if label != query_label else 0 for label in self.labels])).to(self.device) # filter
            return torch.argmin(score).item(), torch.argmax(score).item()
    
    def regen_handler(self, model, loss):
        self.losses.append(loss.item())
        cur_loss = sum(self.losses[-10:])
        self.best_loss = max(self.best_loss, cur_loss)
        if len(self.losses) <= 10:
            logging.info(f"Loss {len(self.losses)}: {loss.item()}")
        if len(self.losses) >= 10 and cur_loss < self.best_loss - 0.2*10:
            self.losses.clear()
            self.best_loss = 0
            self.gen_matrix(model)

class GlobalHardSiameseDataset(torch.utils.data.Dataset):
    def __init__(self, device):
        self.diff_ordered = True
        self.device = device
        self.use_type = torch.half if self.device != torch.device('cpu') else torch.float
        self.gen_batch = 128 if constants.ENV != "local" else (128 if torch.cuda.is_available() and self.device == torch.device('cuda') else 32)
        converter = ConvNeXt_Tiny_Weights.IMAGENET1K_V1.transforms()
        # Processing train data
        df = pd.read_csv(os.path.join(constants.TRAIN_PATH, '.csv'))
        # 25000*3*224*224*16b = 7.53gb
        self.inputs = torch.zeros((len(df), 3, 224, 224), dtype=self.use_type)
        for idx, image_id in tqdm(enumerate(df['Image']), total=len(df['Image'])):
            with Image.open(os.path.join(constants.TRAIN_PATH, image_id)) as im:
                self.inputs[idx] = converter(im.convert("RGB"))
        self.gen_dataset = GenDataset(self.inputs)
        self.loader = torch.utils.data.DataLoader(self.gen_dataset, batch_size=self.gen_batch, shuffle=False)
        # Generate labels
        label_set = set(df['Id'])
        label_set.discard('new_whale')
        label_set = sorted(label_set) # sorted to get a deterministic result
        self.label_dict = {id: idx for idx, id in enumerate(label_set)}
        self.label_dict.update({'new_whale': len(label_set)}) # ensure new_whale is last dimension
        self.nlabel = len(self.label_dict)
        self.labels = np.asarray([self.label_dict[id] for id in df['Id']], dtype=np.short)
        # regen handler vars
        self.losses = []
        self.best_loss = 0
        # global hard vars
        self.prod = []
        self.order = []
        self.pq = []
        
    def __getitem__(self, index):
        idxs = list(self.query_hard())
        return tuple([self.inputs[idx].float() for idx in idxs] + idxs)

    def __len__(self):
        return self.nlabel-1
    
    def get_val(self, triple):
        return (self.prod[triple[0]][self.order[triple[0]][triple[1]]]-self.prod[triple[0]][self.order[triple[0]][-1-triple[2]]], triple)
    
    def get_idx(self, triple):
        return (triple[0], self.order[triple[0]][triple[1]], self.order[triple[0]][-1-triple[2]])
    
    def insert(self, triple):
        self.pq.put(self.get_val(triple))
    
    def gen_prod(self, model): # separate function to deallocate temp vars to reduce RAM usage
        pbar = tqdm(enumerate(self.loader), total=len(self.loader))
        model = model.to(self.device)
        embed_matrix = torch.zeros((len(self.labels), 500), dtype=self.use_type, device=self.device)
        start_idx = 0
        model.eval()
        # gen all output embeddings
        with torch.no_grad():
            for it, x in pbar:
                x = x.to(self.device)
                y = model(x)
                embed_matrix[start_idx:start_idx+x.size(dim=0)] = y.to(dtype=self.use_type)
                start_idx += x.size(dim=0)
                pbar.set_description(f"iter {it}")
        model.train()
        # global hard gen
        # 25000*25000*16b = 1.28gb
        self.prod = (embed_matrix @ embed_matrix.T).to(device='cpu').numpy() # error < 1e-3 (1^2-(2^(-1)*(1+1023/1024))^2 = 9.76e-4)
        # 1.28gb
        self.prod += (self.labels.reshape((-1, 1)) != self.labels.reshape((1, -1))) * np.half(2)

    def gen_matrix(self, model):
        del self.prod
        del self.order
        del self.pq
        self.gen_prod(model)
        # short term 2.5gb? (32bits), long term 8bits, 1.25gb
        self.order = np.argsort(self.prod).astype(np.short)
        self.pq = PriorityQueue()
        for i in range(self.prod.shape[0]):
            self.insert((i,0,0))
    
    def query_hard(self):
        with torch.no_grad():
            item = self.pq.get(block=False)[1]
            self.insert((item[0], item[1]+1, item[2]))
            if item[1] == 0:
                self.insert((item[0], item[1], item[2]+1))
            return self.get_idx(item)
    
    def regen_handler(self, model, loss):
        # track loss, and if too low, regenerate embeddings
        self.losses.append(loss.item())
        cur_loss = sum(self.losses[-10:])
        self.best_loss = max(self.best_loss, cur_loss)
        if len(self.losses) <= 10:
            logging.info(f"Loss {len(self.losses)}: {loss.item()}")
        if len(self.losses) >= 10 and cur_loss < self.best_loss - 0.2*10:
            self.losses.clear()
            self.best_loss = 0
            self.gen_matrix(model)