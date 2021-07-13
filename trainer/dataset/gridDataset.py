from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from glob import glob
import os
import random

class gridDatasetTrain(Dataset):
    def __init__(self, path, transformers):
        super().__init__()
        self.files = sorted(glob(os.path.join(path, '*/*.npy')) * 5)
        self.transformers = transformers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = 1 if os.path.basename(os.path.dirname(self.files[idx])) == 'normal' else 0
        img = np.load(self.files[idx])
        length, w, h = img.shape

        sampling_idx = sorted(random.sample(range(0, length), 9))
        img = img[sampling_idx].reshape((1, 3 * w, 3 * h))
        img = torch.from_numpy(img)
        if self.transformers is not None:
            img = self.transformers(img)

        return (img, label)

class gridDatasetTest(Dataset):
    def __init__(self, path, transformers):
        super().__init__()
        self.files = sorted(glob(os.path.join(path, '*/*.npy')))
        self.transformers = transformers

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label = 1 if os.path.basename(os.path.dirname(self.files[idx])) == 'normal' else 0
        img = np.load(self.files[idx])
        length, w, h = img.shape
        img = img.reshape((1, 3 * w, 3 * h))
        img = torch.from_numpy(img)
        if self.transformers is not None:
            img = self.transformers(img)
        
        return (img, label)

if __name__ == '__main__':
    ds = gridDatasetTrain('/home/cvip/jun/kidnap/dataset/fps2/train')

    dataloader = DataLoader(ds, batch_size=1, shuffle=True)
    print("Train")
    for step, (img, target) in enumerate(dataloader):
        print(f'[{step}] img: {img}')
        print(f'[{step}] img_shape: {img.shape}')
        print(f'[{step}] target: {target}')
        

        if step == 3:
            break

    ds = gridDatasetTrain('/home/cvip/jun/kidnap/dataset/fps2/test')

    dataloader = DataLoader(ds, batch_size=1, shuffle=False)
    print("Test")
    for step, (img, target) in enumerate(dataloader):
        print(f'[{step}] img: {img}')
        print(f'[{step}] img_shape: {img.shape}')
        print(f'[{step}] target: {target}')
        

        if step == 3:
            break