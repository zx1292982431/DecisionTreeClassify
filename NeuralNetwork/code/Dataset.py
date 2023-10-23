import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import os

class MyDataset(Dataset):
    def __init__(self,dataset,data_path):
        self.dataset = dataset
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.data = pd.read_csv(self.data_path,header=None)
        if dataset == 'iris':
            self.data[4] = self.label_encoder.fit_transform(self.data[4])
            self.data = np.array(self.data)
            self.data = torch.tensor(self.data).to(torch.float32)
        elif dataset == 'adult':
            text_cols = [1,3,5,6,7,8,9,13,14]
            for i in text_cols:
                self.data[i] = self.label_encoder.fit_transform(self.data[i])
            self.data = np.array(self.data)
            self.data = torch.tensor(self.data).to(torch.float32)
        elif dataset == 'car':
            text_cols = [0,1,2,3,4,5,6]
            for i in text_cols:
                self.data[i] = self.label_encoder.fit_transform(self.data[i])
            self.data = np.array(self.data)
            self.data = torch.tensor(self.data).to(torch.float32)
        elif dataset == 'wine':
            self.data[0] = self.label_encoder.fit_transform(self.data[0])
            self.data = np.array(self.data)
            self.data = torch.tensor(self.data).to(torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.dataset == 'wine':
            x = self.data[idx,:-1]
            y = self.data[idx,-1]
        else:
            x = self.data[idx,1:]
            y = self.data[idx,0]
        return x,y