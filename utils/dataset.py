from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import savemat, loadmat
import glob
import os
import numpy as np
from utils.utilis import generate_K1map
import torch
import matplotlib.pyplot as plt
class PoissonData(Dataset):
    def __init__(self, path,norm=True, is_training = True, alpha = 10):
        self.path = path
        self.norm = norm
        self.is_training=True
        self.ALPHA = alpha
        try:
            f = []
            for p in path if isinstance(path, list) else [path]:
                p = Path(p)
                if p.is_dir():
                    f += glob.glob(str(p/'**'/"*.*"),recursive=True)
                else:
                    raise Exception(f'{p} does not exist')
            self.mat_files = sorted([x.replace('/',os.sep) for x in f if x.split('.')[-1].lower()=='mat'])
            assert self.mat_files, f'{path}{p}No mat found'
        except Exception as e:
            raise Exception(f'Error loading data from {path}:{e}\n')
    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        path = self.mat_files[idx]
        try:
            struct = loadmat(path)
            y  = struct['data'].astype(np.float32)
            label = struct['label'].astype(np.float32)
            otf3d = struct['otf3d'].astype(np.complex64)
            if self.is_training:
                y = np.random.poisson(np.maximum(self.ALPHA*y,0))
                y = np.asarray(y,dtype=np.float32)
                return torch.from_numpy(y).unsqueeze(0), torch.from_numpy(label), torch.from_numpy(otf3d)
            else:
                return torch.from_numpy(y).unsqueeze(0), torch.from_numpy(label), torch.from_numpy(otf3d)
        except:
            print("Cannot load the .mat file")
            raise ValueError



def create_dataloader_Poisson(path, batch_size, alpha=10, is_training = True):
    dataset = PoissonData(path,alpha=alpha,is_training=is_training)
    batch_size = min(batch_size,len(dataset))
    loader = torch.utils.data.DataLoader
    dataloader = loader(dataset,batch_size,shuffle=False)
    return dataloader,dataset




if __name__ == "__main__":
    path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/data/LLParticle/Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm"
    dataloader, dataset = create_dataloader_Poisson(path,batch_size=2, alpha=30, is_training=True)
    for batch_i, (y, label, otf3d) in enumerate(dataloader):
        break
