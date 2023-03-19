"""
settting:
- Python 3.8.13
- torch 1.11.0
- cuda 11.3
- cudnn 8.2

major difference:
the update sequence becomes
initialization ==> (z-update ==> phi-update ==> x-update) * 5 => output x
change the denoiser part with Unet and also rearrange the depth channel into batch [B*D, 1, H, W]
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utilis import batch_FT2d,batch_iFT2d,autopad
from utils.dataset import create_dataloader_qis
from torch.fft import fft2,ifft2,fftshift,ifftshift
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value,plotcube,plot_img,inner_check_vis
import os
from model.ResUNet import ResUNet


class PLholonetU_block(nn.Module):
    def __init__(self, verboseFlag = False):
        super(PLholonetU_block, self).__init__()
        self.rho1 = torch.nn.Parameter(torch.tensor([1.0],requires_grad=True))
        # torch.nn.init.normal_(self.rho1)
        self.rho2 = torch.nn.Parameter(torch.randn([1]),requires_grad=True)
        torch.nn.init.normal_(self.rho2)
        self.denoiser = ResUNet() # 最后一层为linear output 没有加 activation
        self.flag = verboseFlag
        self.Activation = torch.nn.Sigmoid()

    def batch_forward_proj(self,field_batch, otf3d_batch, intensity=True,scale = True):
        '''

        :param field_batch:  3d field for batch [B, C, H, W]
        :param otf3d_batch: OTF3d for batch [B, C, H, W] or [C, H, W]
        :return: holo_batch [B,1, H, W]
        '''
        [B, C, H, W] = field_batch.shape
        if len(otf3d_batch.shape) == 3:
            otf3d_batch = otf3d_batch.unsqueeze(0)
        assert otf3d_batch.shape[1] == field_batch.shape[1],"The depth slice does not match between field and OTF"
        Fholo3d =  torch.mul(fft2(field_batch),otf3d_batch)
        Fholo = torch.sum(Fholo3d,dim = 1,keepdim=True)
        holo = ifft2(Fholo)
        if intensity:
            holo = torch.abs(holo)
            if scale:
                # max_range = C
                # holo = holo/max_range
                # assert holo.max()[0]< 1.0,"The inner hologram is larger than 1"
                mintmp = holo.view([B,1,H*W]).min(2,keepdim=True)[0].unsqueeze(-1)
                maxtmp = holo.view([B,1,H*W]).max(2,keepdim=True)[0].unsqueeze(-1)
                holo = (holo-mintmp)/(maxtmp-mintmp)
                return holo
            else:
                return holo
        return holo

    def batch_back_proj(self, holo_batch,otf3d_batch,real_constraint= True):
        """

        :param holo_batch: holo_batch [B,1, H, W] or [B,H,W]
        :param otf3d_batch: OTF3d for batch [B, C, H, W] or [C, H, W]
        :return:
        """
        if len(holo_batch.shape) == 3:
            holo_batch = holo_batch.unsqueeze(1) #[B,1,H,W]
        if len(otf3d_batch.shape) == 3:
            otf3d_batch = otf3d_batch.unsqueeze(0)
        holo_batch = holo_batch.to(torch.complex64)
        conj_otf3d = torch.conj(otf3d_batch)
        volumne_slice = otf3d_batch.shape[1]
        holo_expand = holo_batch.tile([1,volumne_slice,1,1])
        holo_expand = fft2(holo_expand) #不需要shift 因为otf3d在构造的时候已经考虑了
        field_ft = torch.multiply(holo_expand,conj_otf3d)
        field3d = ifft2(field_ft) # 不需要shift
        if real_constraint:
            return torch.real(field3d)
        return field3d


    def X_update(self, phi, z, u1, u2, otf3d):
        "proximal operator for forward propagation "
        x1 = phi+u1
        x2 = z+u2
        #numerator n = F(ifft(rho1 * otf * fft(x1)) + rho2*x2)
        temp = self.batch_back_proj(x1, otf3d)
        n = self.rho1*temp+self.rho2*x2
        n = batch_FT2d(n.to(torch.complex64))

        #denominator d = (|OTF|^2 + 1)

        # denominator d = (rho1*|OTF|^2 + rho2)
        #  in fact |OTF|^2==1
        otf_square = torch.abs(otf3d)**2
        ones_array = torch.ones_like(otf_square)
        d =  ones_array*self.rho2+otf_square*self.rho1
        d = d.to(torch.complex64)

        #final fraction
        x_next = batch_iFT2d(n/d)
        return x_next.real

    def Phi_update(self,x,z,u1,u2,otf3d, K1,K0):
        """
        proximal operator for truncated Poisson signal
        :param x: [B,D,H,W]
        :param z:
        :param u1: [B,1,H,W]
        :param u2: []
        :param otf3d:[B,D,H,W]
        :param K1: [B,1,H,W]
        :return: phi_next [B,1,H,W]
        """
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        phi_tilde = self.batch_forward_proj(x,otf3d)-u1
        # phi_tilde = torch.abs(phi_tilde)
        # K0 = self.K*torch.ones_like(K1) - K1 # number of zero in each pixel

        # func = lambda y: self.alpha/self.K*(K0-K1/(torch.exp(self.alpha/self.K*y)-1))+self.rho1*(y-phi_tilde)
        func = lambda y: K0-K1/(torch.exp(y)-1)+self.rho1*(y-phi_tilde)

        # when K1 !=0 solve the one-dimensional equation
        phimin = 1e-5*torch.ones_like(phi_tilde, dtype=phi_tilde.dtype)
        phimax = 100*torch.ones_like(phi_tilde, dtype=phi_tilde.dtype)
        phiave = (phimin + phimax) / 2.0

        for i in range(30):
            tmp = func(phiave)
            ind_pos = tmp>0
            ind_neg = tmp<0

            phimin[ind_neg] = phiave[ind_neg]
            phimax[ind_pos] = phiave[ind_pos]
            phiave = (phimin+phimax)/2.0

        phi_next= phiave
        return phi_next


    def Z_update(self,x,phi,u1,u2,otf3d):
        [B,C,W,H] = x.shape
        z_tilde = x-u2
        z_tilde = z_tilde.view([B*C,1,W,H])
        z_next = self.denoiser(z_tilde)
        z_next = self.Activation(z_next)
        return z_next.view([B,C,W,H])

    def forward(self,x,phi,z,u1,u2,otf3d, K1, K0 ):
        z = self.Z_update(x,phi,u1,u2,otf3d)
        phi = self.Phi_update(x, z, u1, u2, otf3d, K1,K0)
        x = self.X_update(phi, z, u1, u2, otf3d)
        # Lagrangian updates
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        u1 = u1 + phi - self.batch_forward_proj(x,otf3d)
        u2 = u2 +  z - x
        # print(stage_symloss.shape)
        return x,phi,z,u1,u2


class PLholonetU(nn.Module):
    def __init__(self,n,d,sysloss_param = 2e-3,verboseFlag = False):
        super(PLholonetU, self).__init__()
        self.n = n
        self.blocks = nn.ModuleList([])
        for i in range(n):
            self.blocks.append(PLholonetU_block())
        self.Batchlayer = torch.nn.BatchNorm2d(d)
        self.Activation = torch.nn.Sigmoid()
        self.flag = verboseFlag
        if self.flag:
            self.iter = 0
            self.dir = './vis_for_check/'
            if not os.path.isdir(self.dir):
                os.makedirs(self.dir)

    def forward(self, K1,K0, otf3d):
        """
        :param K1: Number of ones in each pixel shape [B,1,W/K,H/K]
        :return:
        """
        # initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = K1.device

        x = self.blocks[0].batch_back_proj(K1,otf3d)

        phi = Variable(K1.data.clone()).to(device)
        z = Variable(x.data.clone()).to(device)
        u1 =torch.zeros(K1.size()).to(device)
        u2 = torch.zeros(x.size()).to(device)

        # building the blocks
        for i in range(self.n):
            x,phi,z,u1,u2 = self.blocks[i](x,phi,z,u1,u2,otf3d,K1,K0) #x,phi,z,u1,u2,otf3d, K1

        x = self.Batchlayer(x)
        x = self.Activation(x)
        # self.iter += 1
        return x

# class denoiser(nn.Module):
#     def __init__(self,fn):
#         super(denoiser,self).__init__()
#         self.CBL_f1 = CBL(1, fn,k=3,s=1,activation=True)
#         self.CBL_f2 = CBL(fn,fn,k=3,s=1,activation=False)
#         self.soft_threshold = SoftThreshold()
#         self.CBL_B1 = CBL(fn, fn,k=3,s=1,activation=True)
#         self.CBL_B2 = CBL(fn,1,k=3,s=1,activation=False)
#
#     def forward(self,xin):
#         [B,C,H,W] = xin.shape
#         xin = xin.view(-1,1,H,W)
#         mems = []
#         mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9 ))
#         x = self.CBL_f2(self.CBL_f1(xin))
#         mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
#         x_out = self.soft_threshold(x)
#         mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
#         x_out = self.CBL_B2(self.CBL_B1(x_out))
#         mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
#         x_out = x_out.view(B,C,H,W)
#         mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
#         x_back =  self.CBL_B2(self.CBL_B1(x))
#         mems.append('%.6gG' % (torch.cuda.memory_reserved() / 1E9))
#         stage_symloss = x_back-xin
#         return x_out,stage_symloss

class forward_block(nn.Module):
    def __init__(self,fn):
        super(forward_block, self).__init__()
        self.CBL1 = CBL(1, fn,k=3,s=1,activation=True)
        self.CBL2 = CBL(fn,fn,k=3,s=1,activation=False)
    def forward(self,x):
        return self.CBL2(self.CBL1(x))

class inverse_block(nn.Module):
    def __init__(self,fn):
        super(inverse_block, self).__init__()
        self.CBL1 = CBL(fn, fn,k=3,s=1,activation=True)
        self.CBL2 = CBL(fn,1,k=3,s=1,activation=False)
    def forward(self,x):
        return self.CBL2(self.CBL1(x))

class CBL(nn.Module):
    def __init__(self,c1,c2,k=1,s=1,padding=None,g=1,activation=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,autopad(k,padding),bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU() if activation is True else(activation if isinstance(activation,nn.Module) else nn.Identity())
    def forward(self,x):
        return self.act(self.bn(self.conv(x)))

class SoftThreshold(nn.Module):
    def __init__(self):
        super(SoftThreshold, self).__init__()

        self.soft_thr = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

    def forward(self, x):
        return torch.mul(torch.sign(x),torch.nn.functional.relu(torch.abs(x)-self.soft_thr))

class denoiser(nn.Module):
    def __init__(self,fn):
        super(denoiser, self).__init__()
        self.F = forward_block(fn)
        self.F_inv = inverse_block(fn)
        self.soft_thr = SoftThreshold()

    def forward(self,xin):
        [B,C,H,W] = xin.shape
        xin = xin.view(-1,1,H,W)
        x = self.F(xin)
        x_thr = self.soft_thr(x)
        x_out = self.F_inv(x_thr)
        x_forward_backward = self.F_inv(x)
        stage_symloss = x_forward_backward-xin
        return x_out.view(B,C,H,W), stage_symloss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # K1 = torch.randn([4,1,256,256])
    # otf3d = obj3d(wave_length = 633*nm, img_rows = 256, img_cols=256, slice=5,size = 10*mm, depth = 2*cm).get_otf3d()
    # net = PLholonet(n=2,d=5).to(device)
    # # x,phi,z,u1,u2 = net(K1,otf3d)
    # x, stage_symlosses = net(K1,otf3d)
    path = "/home/zhangyp/PycharmProjects/PLholo/syn_data/data/train_Nz32_Nxy128_kt30_ks2_ppv2e-04~1e-03"
    path = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/Nz25_Nxy64_kt30_ks8_ppv1e-03~2e-03"
    dataloader, dataset = create_dataloader_qis(path, batch_size=2, Kt=30, Ks=8, norm=False)
    model = PLholonetU(n=5, d=25,verboseFlag=True)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model = model.module.to("cuda")
        model.device = torch.device('cuda')
    else:
        model = torch.nn.DataParallel(model)
        model.device = torch.device('cpu')

    for batch_i, (K1_map,K0_map, label, otf3d, y) in enumerate(dataloader):
        K1_map = K1_map.to(torch.float32).to(device=model.device)
        K0_map = K0_map.to(torch.float32).to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        print("Usage before model",mem)
        x = model(K1_map,K0_map,otf3d)
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        print("Usage after model:", mem)
        break
