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
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value,plotcube,process_cube,get_filtered_centroids
from utils.evaluation_utils import compare_cube
def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))
def psnr(x,im_orig):
    x = norm_tensor(x)
    im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0)* torch.log10(1/ mse)
    return psnr

def render_2d(pred_ctr,gt_ctr,sz=25,label=['pred','gt']):
    fig = plt.figure()
    ax =fig.add_subplot(111)
    ax.scatter(pred_ctr[:,2],pred_ctr[:,1], c='white',edgecolors='g',marker='o',s =sz,label=label[0])
    ax.scatter(gt_ctr[:,2],gt_ctr[:,1], c='white',edgecolors='r',marker='^',s =sz,label=label[1])
    ax.set_title('xy projection')

    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111)
    ax2.scatter(pred_ctr[:,0],pred_ctr[:,1], c="white",edgecolors='g',marker='o',s =sz,label=label[0])
    ax2.scatter(gt_ctr[:,0],gt_ctr[:,1], c='white',edgecolors='r',marker='^',s =sz,label=label[1])
    ax2.set_title('yz projection')

    return ax,ax2
ALPHA = 20
data_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/data/LLParticle/train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm"
model_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/experiment/PLHolo_train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm_L5_B32_lr0.001_Gamma1_Alpha20/2023-03-21-10_59_33/best.pt"
# model_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/experiment/PLHolo_train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm_L5_B32_lr0.001_Gamma1_Alpha20/2023-03-20-14_29_01/best.pt"
# model_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/experiment/PLHolo_train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm_L5_B32_lr0.001_Gamma1_Alpha10/2023-03-21-07_55_52/last.pt"
dataloader, dataset = create_dataloader_Poisson(data_path, batch_size=1, alpha=ALPHA, is_training=True)


Nd = 5
Nz = 7

gamma = 1
model = PLholonet(n=Nd, d=Nz, alpha=ALPHA, sysloss_param=gamma)
if torch.cuda.is_available():
    state_dict = torch.load(model_path,map_location='cuda')
    model = torch.nn.DataParallel(model)
    model = model.module.to("cuda")
    model.device = torch.device('cuda')
else:
    state_dict = torch.load(model_path,map_location='cpu')
    model.device = torch.device('cpu')
model.load_state_dict(state_dict['param'])
y, label, otf3d = next(iter(dataloader))
model.eval()
with torch.no_grad():
        y = y.to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        x, _sloss = model(y, otf3d)
        if len(x.shape) == 4:
            x = x[0, :, :, :]
            label = label[0, :, :, :]
        pred_cube = torch.zeros_like(x)
        idx = (x >= 0.5)
        pred_cube[idx] = torch.ones_like(x)[idx]
        pred_cube = tensor2value(pred_cube)
        gt = tensor2value(label)
        [recall, acc] = prediction_metric(1 - pred_cube, gt, buffer=10, threshold=0.5, grouped=False)
        pred_cube_f,pred_cts = process_cube(1 - pred_cube,radius=5,n_p =5)
        _,gt_cts = process_cube(gt,radius=5,n_p =5)
        # plotcube(gt, 'GT', "Gt.png", show=True)
        # plotcube(pred_cube, 'P_A%.3f_R%.3f' % (acc,recall), "pred.png", show=True)
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        # ax = compare_cube(gt,gt,15,ax=ax,sz=20)
        ax = compare_cube(1 - pred_cube,1 - pred_cube,15,ax=ax,sz=20)
        # ax = compare_cube(1 - pred_cube,gt,15,ax=ax,sz=20)
        # ax.tick_params(axis='z', which='major', labelsize=17)
        # ax.set_title('Precision: %.3f Recall:%.3f'%(recall,acc))
        # ax.legend(["correct","wrong","missed"],loc = 'upper center',ncol= 3,fontsize=10)
        plt.show()
        render_2d(pred_cts,gt_cts,label=['pred','GT'])