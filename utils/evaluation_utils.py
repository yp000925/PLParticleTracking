import torch
import torch.nn as nn
import numpy as np
import random
from torch.fft import fft2,ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt
import math
import scipy.io
from utils.utilis import group_gt
# %%-------------------------------------- Post processing for evaluation  ------------------------------------------

def generate_buffer_field(gt,buffer,grouped=True,**kwargs):
    '''
    :param gt: labelled field with particle center 1 and other pixels 0
    :param buffer: range for buffer in pixel unit
    :return: gt with buffer range
    '''
    [D,H,W] = gt.shape

    if grouped:
        gt,_ = group_gt(gt,**kwargs)

    centers = np.where(gt>0.5) #return tuple
    out = np.zeros_like(gt)
    Y = np.linspace(0,H-1,H)
    X = np.linspace(0,W-1,W)
    yv,xv = np.meshgrid(X,Y)
    for i in range(len(centers[0])):
        d = centers[0][i]
        x = xv-centers[1][i]
        y = yv-centers[2][i]
        dist_sq = x**2+y**2
        temp = np.zeros([H,W])
        temp[dist_sq<=buffer] = 1
        out[d,:,:] += temp
    out[out>0] = np.ones_like(out)[out>0]
    return out,gt
# %%-------------------------------------- Display ------------------------------------------

def volume_render(ctrs,xydim=128,depth=7,file_name=None,show=False, **kwargs):
    '''

    :param ctrs: [z,x,y] for particle centers
    :param kwargs:
    :return:
    '''
    z_tp = ctrs[:,0]
    x_tp = ctrs[:,1]
    y_tp = ctrs[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(x_tp,y_tp,z_tp,c='g',marker='o',s =70)
    ax.set_xlim([0,xydim])
    ax.set_ylim([0,xydim])
    ax.set_zlim([0,depth])
    ax.tick_params(axis='both', which='major', labelsize=12)
    if file_name:
        plt.savefig(file_name)
    if show:
        plt.show()


def plotcube(vol, fig_title, file_name=None, show=True,**kwargs):
    # maxval = np.amax(vol)
    # minval = np.amin(vol)
    # vol = (vol - minval) / (maxval - minval+1e-7)

    Nz, Nx, Ny = np.shape(vol)

    if Nz <= 10:
        img_col_n = Nz
    else:
        img_col_n = math.ceil(np.sqrt(Nz))

    img_row_n = math.ceil(Nz / img_col_n)
    image_height = 5
    fig = plt.figure(figsize=(img_col_n * image_height, image_height * img_row_n + 0.5))
    fig.suptitle(fig_title, y=1)
    img_n = 0
    for iz in range(Nz):
        img_n = img_n + 1
        ax = fig.add_subplot(img_row_n, img_col_n, img_n)
        ax.set_title("z " + str(img_n))
        slice = vol[iz, :, :]
        im = ax.imshow(slice, aspect='equal',**kwargs)

    fig.tight_layout()
    if file_name:
        plt.savefig(file_name)
    if show:
        plt.show()


def plotcubic_centers(coords, fig_title, file_name=None, show=True, vol_shape=[7,512,512],**kwargs):
    """
    :param coords: [x,y,z] of N particles centroids
    :param fig_title:
    :param file_name:
    :param show:
    :param kwargs:
    :return:
    """
    Nz, Nx, Ny = vol_shape
    Nc, _ = np.shape(coords)

    if Nz <= 10:
        img_col_n = Nz
    else:
        img_col_n = math.ceil(np.sqrt(Nz))

    img_row_n = math.ceil(Nz / img_col_n)
    image_height = 5
    fig = plt.figure(figsize=(img_col_n * image_height, image_height * img_row_n + 0.5))
    fig.suptitle(fig_title, y=1)
    img_n = 0
    for iz in range(Nz):
        coord = coords[coords[:,0]==iz]
        img_n = img_n + 1
        ax = fig.add_subplot(img_row_n, img_col_n, img_n)
        ax.set_title("z " + str(img_n))
        slice=np.zeros([Nx,Ny])
        slice[coord[:,1],coord[:,2]] = np.ones_like(slice)[coord[:,1],coord[:,2]]
        im = ax.imshow(slice, aspect='equal',**kwargs)
    fig.tight_layout()
    if file_name:
        plt.savefig(file_name)
    if show:
        plt.show()

def compare_cube(pred,gt,buffer=None,grouped=False,ax =None,sz = 25):
    mask,_ = generate_buffer_field(gt,buffer,grouped)
    TP = mask * pred
    FP = (1-mask)* pred
    mask2,_ = generate_buffer_field(pred,buffer,grouped=False)
    FN =  (1-mask2)*gt# approximate missed particles
    z_tp,x_tp,y_tp = np.where(TP>0.5)
    z_fp,x_fp,y_fp = np.where(FP>0.5)
    z_fn,x_fn,y_fn= np.where(FN>0.5)
    if not ax:
        fig = plt.figure()
        ax =fig.add_subplot(111,projection = '3d')
    ax.scatter(x_tp,y_tp,z_tp,c='g',marker='o',s =sz)
    ax.scatter(x_fp,y_fp,z_fp,c='r',marker='o',s =sz)
    ax.scatter(x_fn,y_fn,z_fn,c='y',marker='^',s =sz)

    return ax

if __name__=="__main__":

    pred = np.zeros((2,3,3))
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    [z,x,y] = pred
    ax.scatter(x_tp,y_tp,z_tp,c='g',marker='o')
    ax.scatter(x_fp,y_fp,z_fp,c='r',marker='^')
    ax.scatter(x_fn,y_fn,z_fn,c='y',marker='o')
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)