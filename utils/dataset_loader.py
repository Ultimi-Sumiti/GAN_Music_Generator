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
        return self.file.shape

    def __getitem__(self, idx):
        matrix = self.file[idx]
        return torch.tensor(matrix)


# Define the meastro DataLoader.
class MaestroV3DataModule(L.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 32):
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.dataset = MaestroV3DataSet(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


# Small tester.
if __name__ == "__main__":
    dataset_path = "../data/preprocessed/maestro-v3.0.0/dataset1/d1.h5"

    dataset = MaestroV3DataSet(dataset_path)
    print("Dataset size =", dataset.__len__())

    sample = dataset[0]
    print(sample.shape)

    print((sample > 0).sum())
    
