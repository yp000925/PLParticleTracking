"""
settting:
- Python 3.8.13
- torch 1.11.0
- cuda 11.3
- cudnn 8.2

This is for the model of phase retrieval -> phase range is [-pi, pi]
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.utilis import batch_FT2d,batch_iFT2d,autopad
from utils.PhaseDataset import create_dataloader_qis
from torch.fft import fft2,ifft2,fftshift,ifftshift
import os
from model.ResUNet import ResUNet


class PLPhaseNetU_block(nn.Module):
    def __init__(self, verboseFlag = False):
        super(PLPhaseNetU_block, self).__init__()

        self.rho1 = torch.nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        torch.nn.init.normal_(self.rho1)
        self.rho2 = torch.nn.Parameter(torch.randn([1]),requires_grad=True)
        torch.nn.init.normal_(self.rho2)
        # self.lamda = torch.nn.Parameter(torch.randn([1]),requires_grad=True).to(device)
        # torch.nn.init.normal_(self.lamda)
        #self.denoiser = ResUNet().to(device)
        self.denoiser = ResUNet()
        self.flag = verboseFlag

    def batch_forward_proj(self,obj, otf, intensity=True, scale = True):
        '''

        :param obj:  phasemap [B, 1, H, W] within range [0,1]
        :param otf: OTF for batch [B,1, H, W]
        :return: holo_batch [B, 1, H, W]
        '''
        [B, _, H, W] = obj.shape
        # phase = torch.exp(1j*torch.pi * obj)
        phase = torch.exp(1j*(-torch.pi+2*torch.pi * obj)) ## mainly change point
        Fholo  =  torch.mul(fft2(phase),otf)
        holo = ifft2(Fholo)
        if intensity:
            holo = torch.square(torch.abs(holo))
            if scale:
                # max_range = C
                # holo = holo/max_range
                # assert holo.max()[0]< 1.0,"The inner hologram is larger than 1"
                mintmp = holo.view([B,1,H*W]).min(2,keepdim=True)[0].unsqueeze(-1)
                maxtmp = holo.view([B,1,H*W]).max(2,keepdim=True)[0].unsqueeze(-1)
                # holo = (holo-mintmp)/(maxtmp-mintmp)
                holo = holo/maxtmp
                return holo
            else:
                return holo
        return holo

    def batch_back_proj(self, holo_batch, otf, phase_clipped= True):
        """
        :param holo_batch: holo_batch [B, 1, H, W]
        :param otf3d_batch: OTF3d for batch [B,1, H, W]
        :phase_clipped Ture: constrain the phase within [0,1] for [0,pi]
        :return:
        """
        holo_batch = holo_batch.to(torch.complex64)
        conj_otf3d = torch.conj(otf)
        holo_expand = fft2(holo_batch) # 不需要shift 因为otf3d在构造的时候已经考虑了
        field_ft = torch.multiply(holo_expand,conj_otf3d)
        field = ifft2(field_ft) # 不需要shift
        if phase_clipped:
            return torch.abs(torch.angle(field)/torch.pi)
        return field

    def X_update(self, phi, z, u1, u2, otf3d):
        "proximal operator for forward propagation "
        x1 = phi + u1 #holo map consistance
        x2 = z + u2 #phase map consistance
        #numerator n = F(ifft(rho1 * otf * fft(x1)) + rho2*x2)
        temp = self.batch_back_proj(x1, otf3d) # already perform the ifft
        n = self.rho1*temp+self.rho2*x2
        n =batch_FT2d(n.to(torch.complex64))

        # denominator d = (rho1*|OTF|^2 + rho2)
        #  in fact |OTF|^2==1
        otf_square = torch.abs(otf3d)**2
        ones_array = torch.ones_like(otf_square)
        d =  ones_array*self.rho2+otf_square*self.rho1
        d = d.to(torch.complex64)

        #final fraction
        x_next = batch_iFT2d(n/d)

        x_next = torch.abs(x_next)
        [B, _, H, W] = x_next.shape
        maxtmp = x_next.view([B,1,H*W]).max(2,keepdim=True)[0].unsqueeze(-1)
        return x_next/maxtmp

    def Phi_update(self,x,z,u1,u2,otf, K1,K0):
        """
        proximal operator for truncated Poisson signal
        :param x: [B,1,H,W] phasemap with range [0,1]
        :param z:
        :param u1: [B,1,H,W]
        :param u2: []
        :param otf3d:[B,D,H,W]
        :param K1: [B,1,H,W]
        :return: phi_next [B,1,H,W]
        """
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        phi_tilde = self.batch_forward_proj(x,otf)-u1


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
        z_tilde = x-u2
        # z_tilde = z_tilde.view([B*C,1,W,H])
        z_next = self.denoiser(z_tilde)
        return z_next

    def forward(self,x,phi,z,u1,u2,otf3d, K1,K0):
        # t0 = time.time()
        # U, Z and X updates
        z = self.Z_update(x,phi,u1,u2,otf3d)
        phi = self.Phi_update(x, z, u1, u2, otf3d, K1,K0)
        x = self.X_update(phi, z, u1, u2, otf3d)
        # t1 = time.time()
        # print(t1-t0,x.shape,phi.shape,z.shape,u1.shape,u2.shape)

        # t2 = time.time()
        # print(t2-t1,x.shape,phi.shape,z.shape,u1.shape,u2.shape)

        # t3 = time.time()
        # print(t3-t2,x.shape,phi.shape,z.shape,u1.shape,u2.shape)
        # Lagrangian updates
        # batch_size = x.shape[0]
        # otf3d_tensor = otf3d.tile([batch_size,1,1,1])
        u1 = u1 + phi - self.batch_forward_proj(x,otf3d)
        u2 = u2 +  z - x
        # print(stage_symloss.shape)
        return x,phi,z,u1,u2


class PLPhasenetU(nn.Module):
    def __init__(self,n,sysloss_param = 2e-2,verboseFlag = False):
        super(PLPhasenetU, self).__init__()
        self.n = n
        self.blocks = nn.ModuleList([])
        for i in range(n):
            self.blocks.append(PLPhaseNetU_block())
        self.Batchlayer = torch.nn.BatchNorm2d(1)
        self.Activation = torch.nn.LeakyReLU()
        self.sysloss_param = sysloss_param
        # if self.flag:
        #     self.iter = 0
        #     self.dir = './vis_for_check/'
        #     if not os.path.isdir(self.dir):
        #         os.makedirs(self.dir)

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
        stage_symlosses = torch.tensor([0.0]).to(device)

        for i in range(self.n):
            x,phi,z,u1,u2 = self.blocks[i](x,phi,z,u1,u2,otf3d,K1,K0) #x,phi,z,u1,u2,otf3d, K1
            # print('stage',i,'\n symloss',stage_symlosses)

        x = self.Batchlayer(x)
        x = self.Activation(x)
        return x

# class resblock(nn.Module):
#     def __init__(self,c):
#         super(resblock, self).__init__()
#         self.CBL1 = CBL(c, c,k=3,s=1,padding='same',activation=True)
#         self.CBL2 = CBL(c, c,k=3,s=1,padding='same',activation=False)
#         self.act = nn.LeakyReLU()
#
#     def forward(self,x):
#         return self.act(x+self.CBL2(self.CBL1(x)))

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

class denoiser(nn.Module):
    def __init__(self,fn):
        super(denoiser, self).__init__()
        self.F = forward_block(fn)
        self.F_inv = inverse_block(fn)
        self.soft_thr = SoftThreshold()

    def forward(self,xin):
        x = self.F(xin)
        x_thr = self.soft_thr(x)
        x_out = self.F_inv(x_thr)
        x_forward_backward = self.F_inv(x)
        stage_symloss = x_forward_backward-xin
        return x_out, stage_symloss

class CBL(nn.Module):
    def __init__(self,c1,c2,k=3,s=1,padding=None,g=1,activation=True):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(c1,c2,k,s,padding=autopad(k,padding),bias=False)
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



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # K1 = torch.randn([4,1,256,256])
    # otf3d = obj3d(wave_length = 633*nm, img_rows = 256, img_cols=256, slice=5,size = 10*mm, depth = 2*cm).get_otf3d()
    # net = PLholonet(n=2,d=5).to(device)
    # # x,phi,z,u1,u2 = net(K1,otf3d)
    # x, stage_symlosses = net(K1,otf3d)
    path = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/Lambda6.33e-07_Nx256_Ny256_deltaXY8e-06_z0.022_kt30_ks4"
    dataloader, dataset = create_dataloader_qis(path,3,Kt=30,Ks=4,norm=False)
    model = PLPhasenetU(n=5)
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
        x = model(K1_map,K0_map,otf3d)
        break
