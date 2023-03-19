import torch
import numpy as np
# %%
nm = 1e-9
mm = 1e-3
cm = 1e-2
class obj3d():
    def __init__(self,wave_length = 633*nm, img_rows = 64, img_cols=64, slice=5,size = 10*mm, depth = 2*cm):
        self.lamda = wave_length
        self.x = img_cols
        self.y = img_rows
        self.slice = slice
        self.xy_res = size/img_rows
        self.z_res = depth/slice

    def get_otf3d(self):
        k = 2*np.pi/self.lamda
        r1 = torch.linspace(-self.x/2,self.x/2-1,self.x)
        s1 = torch.linspace(-self.y/2,self.y/2-1,self.y)
        deltaFx = 1/(self.x*self.xy_res)*r1
        deltaFy = 1/(self.y*self.xy_res)*s1
        meshgrid_x,meshgrid_y = torch.meshgrid(deltaFx,deltaFy)
        q_term = torch.pow(self.lamda * meshgrid_x, 2) + torch.pow(self.lamda * meshgrid_y, 2)
        q_term = torch.sqrt(torch.ones_like(q_term)-q_term)
        z = np.linspace(0,self.z_res*self.slice,self.slice)
        obj3d = [1j*k*q_term.unsqueeze(0)*x for x in z]
        obj3d = torch.concat(obj3d,dim=0)
        return obj3d


if __name__== "__main__":
    otf3d = obj3d().get_otf3d()