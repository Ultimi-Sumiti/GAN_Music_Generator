import pytorch_lightning as L
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import random
import numpy as np
 

# Define the MaestroV3DataSet - CPU version (no preload).
class MaestroV3DataSet(Dataset):

    def __init__(self, file_path: str, mode: str = "single"):
        # Store the file path
        self.h5_path = file_path
        # Save dataset length.
        with h5py.File(self.h5_path, "r") as f:
            self.length = len(f['x'])
        # Mode can be either single or pair.
        assert mode == "single" or mode == "pair"

        # Save mode.
        self.mode = mode

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as db:
            if self.mode == "single":
                # MODEL 1
                sample = db['x'][idx]
                sample = torch.tensor(sample, dtype=torch.float32) # [128, 16]
                sample = sample.unsqueeze(0) # [1, 128, 16]
                return sample
    
            else:
                # MODEL 2
                prev, curr = db['x'][idx]
    
                prev = torch.tensor(prev, dtype=torch.float32) # [128, 16]
                curr = torch.tensor(curr, dtype=torch.float32) # [128, 16]
    
                prev = prev.unsqueeze(0) # [1, 128, 16]
                curr = curr.unsqueeze(0) # [1, 128, 16]
                
                return prev, curr


# Define the MaestroV3DataSet - GPU version (with preload).
# The entire dataset is entirely preloaded in the GPU.
class MaestroV3DataSet_GPU(Dataset):
    def __init__(self, file_path: str, mode: str = "single"):
        self.mode = mode

        # Open h5 file and save the dataset.
        with h5py.File(file_path, 'r') as f:
            dataset = f['x'][:]

        # Preload dataset in GPU.
        device = "cuda"

        if mode == "single":
            # from [N, 128, 16] to  [N, 1, 128, 16].
            self.dataset = torch.tensor(data, dtype=torch.float32).unsqueeze(1)
            # Move to GPU.
            self.dataset = self.dataset.to(device)
        else:
            prev, curr = zip(*dataset)
            self.dataset = [
                (
                    # from [128, 16] to [1, 128, 16] -> move to GPU.
                    torch.tensor(p, dtype=torch.float32).unsqueeze(0).to(device),
                    # from [128, 16] to [1, 128, 16] -> move to GPU.
                    torch.tensor(c, dtype=torch.float32).unsqueeze(0).to(device)
                )
                for p, c in zip(prev, curr)
            ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Define the meastro DataLoader.
class MaestroV3DataModule(L.LightningDataModule):

    def __init__(
            self, 
            data_dir: str,
            batch_size: int = 32,
            mode: str = "single",
            num_workers: int = 0,
            preload_gpu: bool = False
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.mode = mode
        self.preload_gpu = preload_gpu
        self.nw = num_workers

    def setup(self, stage: str):
        if self.preload_gpu:
            self.dataset = MaestroV3DataSet_GPU(self.data_dir, self.mode)
        else:
            self.dataset = MaestroV3DataSet(self.data_dir, self.mode)

    def train_dataloader(self):
        if self.preload_gpu:
            dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
        else:
            dataloader = DataLoader(
                self.dataset,
                num_workers=self.nw,
                batch_size=self.batch_size,
                shuffle=True
            )
        return dataloader


# Random sampler from dataset.
def random_batch_sampler(dataset: Dataset, size: int):
    prevs = []
    currs = []
    for _ in range(size):
        rnd_idx = random.randint(0, len(dataset)-1)
        sample = dataset[rnd_idx]
        prev, curr = sample
        prevs.append(prev)
        currs.append(curr)

    batch = torch.from_numpy(np.array([prevs, currs]))
    return batch
