import pytorch_lightning as L
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

# Define the MaestroV3DataSet.
class MaestroV3DataSet(Dataset):

    def __init__(self, file_path):
        self.file = h5py.File(file_path, "r")['x']

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        matrix = self.file[idx]
        tensor = torch.tensor(matrix, dtype=torch.float32)
        return tensor.unsqueeze(0)


# Define the meastro DataLoader.
class MaestroV3DataModule(L.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.dataset = MaestroV3DataSet(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)