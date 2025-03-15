import numpy as np
import torch
import pickle
from torch_geometric.data import Batch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data=None, file_path='./data/data.pt', subset_size=None):
        if data is not None:
            self.data = data
        else:
            with open(file_path, 'rb') as f:
                self.data = pickle.load(f)
            if subset_size is not None:
                self.data = self.data[:subset_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data(self, indices):
        subset_data = [self.data[i] for i in indices]
        return MyDataset(data=subset_data)

def collate(data_list):
    d1_list, d2_list, label_list = [], [], []
    for d in data_list:
        graph1, graph2, cell, label = d[0], d[1], d[2], d[3]
        graph1.cell = cell  # Attach cell to graph1 for convenience
        d1_list.append(graph1)
        d2_list.append(graph2)
        label_list.append(label)

    batch1 = Batch.from_data_list(d1_list)
    batch2 = Batch.from_data_list(d2_list)
    return batch1, batch2, torch.tensor(label_list, dtype=torch.long)
