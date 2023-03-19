from torch.utils.data import Dataset
from pathlib import Path
from scipy.io import savemat, loadmat
import glob
import os
import numpy as np
from utils.utilis import generate_K1map
import torch

class load_qis_mat(Dataset):
    def __init__(self, path, Kt, Ks, norm=True):
        self.Kt = Kt
        self.Ks = Ks
        self.path = path
        self.norm = norm
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
            y  = struct['y'].astype(np.float32)
            otf3d = struct['otf3d'].astype(np.complex64)
            if y.shape[-1] == self.Kt:
                y = np.transpose(y,[2,0,1])
            elif y.shape[0] == self.Kt:
                pass
            else:
                raise Exception("The temperal sampling dose not match")
        except:
            print("Cannot load the .mat file")
            raise ValueError
        if self.norm:
            K1_map = generate_K1map(y, [self.Ks,self.Ks], norm=True) # range from 0-1
            return torch.from_numpy(K1_map).unsqueeze(0),torch.from_numpy(otf3d),torch.from_numpy(y),path
        else:
            K1_map =  generate_K1map(y, [self.Ks,self.Ks], norm=False)
            K0_map = self.Ks*self.Ks*self.Kt*np.ones_like(K1_map)-K1_map
            return torch.from_numpy(K1_map).unsqueeze(0), torch.from_numpy(K0_map).unsqueeze(0),torch.from_numpy(otf3d),torch.from_numpy(y),path

class load_qis_mat_lite(Dataset):
    def __init__(self, path, Kt, Ks, norm=True):
        super(load_qis_mat_lite,self).__init__()
        self.Kt = Kt
        self.Ks = Ks
        self.path = path
        self.norm = norm
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
            k1 = struct['k1'].astype(np.float32)
            otf3d = struct['otf3d'].astype(np.complex64)
        except:
            print("Cannot load the .mat file")
            raise ValueError
        if self.norm:
            K1_map = k1/(self.Ks**self.Ks*self.Kt)
            return torch.from_numpy(K1_map).unsqueeze(0), torch.from_numpy(otf3d),path
        else:
            K1_map = k1
            K0_map = self.Ks*self.Ks*self.Kt*np.ones_like(K1_map)-K1_map
            return torch.from_numpy(K1_map).unsqueeze(0), torch.from_numpy(K0_map).unsqueeze(0),torch.from_numpy(otf3d),path


def create_test_dataset(path, batch_size, Kt, Ks, norm=True,lite=False):
    if lite:
        dataset = load_qis_mat_lite(path,Kt,Ks,norm=norm)
    else:
        dataset = load_qis_mat(path,Kt,Ks,norm=norm)
    batch_size = min(batch_size,len(dataset))
    loader = torch.utils.data.DataLoader
    dataloader = loader(dataset,batch_size,shuffle=False)
    return dataloader,dataset




if __name__ == "__main__":
    path ="/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/realParticle/K1map_Lambda6.6e-07_Nx256_Ny256_deltaXY1.38e-05_kt1600_ks1"
    dataloader, dataset = create_test_dataset(path,batch_size=2,Kt=1600,Ks=1,norm=False,lite=True)
    for batch_i, (K1_map,k0_map, otf3d,p) in enumerate(dataloader):
        break# if __name__ == "__main__":
#     y_pred = torch.zeros([2,3,4,4])
#     a = 2*torch.ones([2,2])
#
#     y_true = torch.ones_like(y_pred)
#     pcc(y_pred,y_true)
