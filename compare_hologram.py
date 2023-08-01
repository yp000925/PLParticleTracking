import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import logging
from model.PLholonet import PLholonet
from utils.dataset import create_dataloader_Poisson
from utils.utilis import PCC, PSNR, accuracy, random_init, tensor2value, plotcube, acc_and_recall_with_buffer, \
    prediction_metric
from torch.optim import Adam
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import re
import matplotlib.pyplot as plt

def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))
def psnr(x,im_orig):
    x = norm_tensor(x)
    im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0)* torch.log10(1/ mse)
    return psnr

path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/data/LLParticle/train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm"
dataloader1, dataset1 = create_dataloader_Poisson(path,batch_size=1, alpha=30, is_training=True)
dataloader2, dataset2 = create_dataloader_Poisson(path,batch_size=1, alpha=20, is_training=True)
dataloader3, dataset3 = create_dataloader_Poisson(path,batch_size=1, alpha=10, is_training=True)
dataloader, dataset = create_dataloader_Poisson(path,batch_size=1, alpha=10, is_training=False)
for batch_i, (y1, label, otf3d) in enumerate(dataloader1):
    break
for batch_i, (y2, label, otf3d) in enumerate(dataloader2):
    break
for batch_i, (y3, label, otf3d) in enumerate(dataloader3):
    break
for batch_i, (y, label, otf3d) in enumerate(dataloader):
    break

psnr1 = psnr(y1,y)
psnr2 = psnr(y2,y)
psnr3 = psnr(y3,y)
