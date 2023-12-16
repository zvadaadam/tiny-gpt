import os
import torch
import requests
import numpy as np
from torch.utils.data import Dataset

class ShakespeareDataset(Dataset):
    
    def __init__(self, train=True):
        
        data = self.download()

        # claculate vocab size - get all the unique characters that occur in this text
        self.chars = self.unique_char(data)
        self.vocab_size = len(self.chars)
        
        print("all the unique characters:", ''.join(self.chars))
        print(f"vocab size: {self.vocab_size:,}")

        # create a mapping from characters to integers
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

        # create the train and test splits
        n = len(data)
        self.train_data = data[:int(n*0.9)]
        self.val_data = data[int(n*0.9):]

        # encode both to integers
        self.train_ids = self.encode(self.train_data)
        self.val_ids = self.encode(self.val_data)
        
        print(f"train has {len(self.train_ids):,} tokens")
        print(f"val has {len(self.val_ids):,} tokens")
        
    def get_batch(self, batch_size: int, block_size: int, split: str):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([torch.tensor(self.encode(data[i:i+block_size])) for i in ix])
        y = torch.stack([torch.tensor(self.encode(data[i+1:i+1+block_size])) for i in ix])
        return x, y
        
    def download(self):
        input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
        if not os.path.exists(input_file_path):
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(input_file_path, 'w') as f:
                f.write(requests.get(data_url).text)

        # read the data
        with open(input_file_path, 'r') as f:
            data = f.read()
        print(f"length of dataset in characters: {len(data):,}")
        
        return data

    def encode(self, s):
        return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers

    def decode(self, l):
        return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

    def unique_char(self, data):
        return sorted(list(set(data)))
    
    
if __name__ == "__main__":
    dataset = ShakespeareDataset()
    print(dataset.vocab_size)
    
    x, y = dataset.get_batch(2, 32, 'train')
    
    print(x.shape)
    print(y.shape)
    
    print(x[0])
    print(dataset.decode(x[0].tolist()))