import pytorch_lightning as L
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py

# Define the MaestroV3DataSet.
class MaestroV3DataSet(Dataset):

    def __init__(self, file_path: str, mode: str = "single"):
        # Save dataset file.
        self.file = h5py.File(file_path, "r")['x']

        # Mode can be either single or pair.
        assert mode == "single" or mode == "pair"

        # Save mode.
        self.mode = mode

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):

        if self.mode == "single":
            # MODEL 1
            sample = self.file[idx]
            sample = torch.tensor(sample, dtype=torch.float32) # [128, 16]
            sample = sample.unsqueeze(0) # [1, 128, 16]
            return sample

        else:
            # MODEL 2
            prev, curr = self.file[idx]

            prev = torch.tensor(prev, dtype=torch.float32) # [128, 16]
            curr = torch.tensor(curr, dtype=torch.float32) # [128, 16]

            prev = prev.unsqueeze(0) # [1, 128, 16]
            curr = curr.unsqueeze(0) # [1, 128, 16]
            
            return prev, curr


# Define the meastro DataLoader.
class MaestroV3DataModule(L.LightningDataModule):

    def __init__(self, data_dir: str, batch_size: int = 32, mode: str = "single"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode

    def setup(self, stage: str):
        self.dataset = MaestroV3DataSet(self.data_dir, self.mode)

    def train_dataloader(self):
        nw = 9 # Shuld be tuned based on the CPU.
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                         num_workers=nw)
