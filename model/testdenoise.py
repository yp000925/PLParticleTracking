import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utilis import batch_FT2d,batch_iFT2d,autopad
from utils.dataset import create_dataloader_qis
from torch.fft import fft2,ifft2,fftshift,ifftshift
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value,plotcube,plot_img,inner_check_vis
import os
from model.PLholonet import CBL, resblock,SoftThreshold

class denoiser1(nn.Module):
    def __init__(self,fn):
        super(denoiser1,self).__init__()
        self.CBL_f1 = CBL(1, fn,k=3,s=1,activation=True)
        self.CBL_f2 = CBL(fn,fn,k=3,s=1,activation=False)
        self.soft_threshold = SoftThreshold()
        self.CBL_B1 = CBL(fn, fn,k=3,s=1,activation=True)
        self.CBL_B2 = CBL(fn,1,k=3,s=1,activation=False)
        self.mems = ['%.6gG' % (torch.cuda.memory_reserved()/ 1E9)]

    def forward(self,xin):
        [B,C,H,W] = xin.shape
        print('denoiser 1 ==================')
        xin = xin.view(-1,1,H,W)
        print('xin shape',xin.shape)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x = self.CBL_f2(self.CBL_f1(xin))
        print('x shape',x.shape)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_out = self.soft_threshold(x)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_out = self.CBL_B2(self.CBL_B1(x_out))
        print('xout shape',x_out.shape)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_out = x_out.view(B,C,H,W)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_back =  self.CBL_B2(self.CBL_B1(x))
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        stage_symloss = x_back-xin
        return x_out,stage_symloss,self.mems

class denoiser2(nn.Module):
    def __init__(self,c):
        super(denoiser2, self).__init__()
        self.resblock1 = resblock(c)
        self.resblock2 = resblock(c)
        self.soft_thr = SoftThreshold()
        self.mems = ['%.6gG' % (torch.cuda.memory_reserved()/ 1E9)]

    def forward(self,xin):
        print('denoiser 2 ==================')
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x = self.resblock1(xin)
        print('xin shape',xin.shape)
        print('x shape',x.shape)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_thr = self.soft_thr(x)
        print('x_thr shape',x_thr.shape)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_out = self.resblock2(x_thr)
        print('xout shape',x_out.shape)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        x_forward_backward = self.resblock2(x)
        self.mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
        stage_symloss = x_forward_backward-xin
        return x_out,stage_symloss,self.mems




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn([64,32,128,128]).to(torch.float32).to(device=device)
    mems = []
    d1 = denoiser1(32)
    d2 = denoiser2(32)
    d1 = torch.nn.DataParallel(d1)
    d1 = d1.module.to("cuda")
    d2 = torch.nn.DataParallel(d2)
    d2 = d2.module.to("cuda")
    mems.append('%.6gG' % (torch.cuda.memory_reserved()/ 1E9))
    x2 = d2(x)
    mems.append('%.6gG' % (torch.cuda.memory_reserved()/ 1E9))
    # x1 = d1(x)
    # mems.append('%.6gG' % (torch.cuda.memory_reserved()/ 1E9))
    # K1 = torch.randn([4,1,256,256])
    # otf3d = obj3d(wave_length = 633*nm, img_rows = 256, img_cols=256, slice=5,size = 10*mm, depth = 2*cm).get_otf3d()
    # net = PLholonet(n=2,d=5).to(device)
    # # x,phi,z,u1,u2 = net(K1,otf3d)
    # x, stage_symlosses = net(K1,otf3d)
