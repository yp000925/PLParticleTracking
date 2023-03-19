'''

'''
import torch
import torch.fft as tfft
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from numpy.fft import fft2, ifft2, ifftshift


class X_Update(nn.Module):
    '''
    The proximator for least square update
    Based on the forward model for gabor holographic imaging
    The input  kernel
    '''
    def __init__(self):
        super(X_Update, self).__init__()
        self.rho1 = torch.nn.Parameter(torch.randn([1]),requires_grad=True)
        torch.nn.init.normal_(self.rho1)
        self.rho2 = torch.nn.Parameter(torch.randn([1]),requires_grad=True)
        torch.nn.init.normal_(self.rho2)

    def back_prop(self,holo,otf3d):
        '''

        :param holo: hologram
        :param otf3d: H(kx,ky,kz)
        :return: Ht*holo
        '''
        volumne_slice = otf3d.shape[1]

        holo = holo.to(torch.complex64)
        otf3d = otf3d.to(torch.complex64)
        conj_otf3d = torch.conj(otf3d)

        # perform iFT(FT(o)*conj(h))
        holo_expand = holo.unsqueeze(1).repeat([1,volumne_slice,1,1])
        holo_expand = torch.fft.fftn(holo_expand,dim =[2,3])
        field_ft = torch.multiply(holo_expand,conj_otf3d)
        field3d = torch.fft.ifftn(field_ft,dim=[2,3])
        return torch.real(field3d)

    def forward(self, x1, x2, otf3d):
        #numerator n = F( alpha * At * I_h + v)
        temp = self.back_prop(x1, otf3d)
        n = self.rho1*temp+self.rho2*x2
        n = torch.fft.fftn(n.to(torch.complex64),dim = [2,3])

        #denominator d = (|OTF|^2 + 1)
        otf_square = torch.abs(otf3d)**2
        ones_array = torch.ones_like(otf_square)
        d =  ones_array*self.rho2+otf_square*self.rho1
        d = d.to(torch.complex64)

        #final fraction
        x_next = torch.fft.ifftn(n/d,dim=[2,3])
        return x_next.real

class V_update(nn.Module):
    def __init__(self,alpha = 8, K = 8):
        super(V_update,self).__init__()

    def forward(self,u, v, K1, K, rho):
        """ proximal operator "prox_{alpha f}" for single photon imaging """
        u = np.array(u)
        v = np.array(v)

        vtilde = v-u
        x = np.copy(vtilde)
        K0 = np.square(K) - K1

        ind0 = (K1==0)
        x[ind0] = vtilde[ind0]-K0[ind0]/rho