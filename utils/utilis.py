import torch
import torch.nn as nn
import numpy as np
import random
from torch.fft import fft2,ifft2,fftshift,ifftshift
import matplotlib.pyplot as plt
import math
import scipy.io


# %%-------------------------------------- Model blocks ------------------------------------------
def random_init(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class resblock(nn.Module):
    def __init__(self,c):
        super(resblock, self).__init__()
        self.CBL1 = CBL(c, c,k=3,s=1,activation=True)
        self.CBL2 = CBL(c, c,k=3,s=1,activation=False)
        self.act = nn.LeakyReLU()

    def forward(self,x):
        return self.act(x+self.CBL2(self.CBL1(x)))


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

def tensor2value(tensor):
    return tensor.data.cpu().numpy()

def batch_FT2d(a_tensor):# by default FFTs the last two dimensions
    assert len(a_tensor.shape)==4, "expected dimension is 4 with batch size at first"
    return ifftshift(fft2(fftshift(a_tensor,dim = [2,3])),dim=[2,3])
#
# def batch_FT2d(a):
#     assert len(a.shape)==4, "expected dimension is 4 with batch size at first"
#     A = torch.zeros(a)
#     centre = a.shape[2]//2 +1
#     A[:, :, :centre, :centre] = a[:, :, (centre-1):, (centre-1):]
#     A[:, :, :centre, -(centre-1):] = a[:, :, (centre-1):, :(centre-1)]
#     A[:, :, -(centre-1):, :centre] = a[:, :, : (centre-1), (centre-1):]
#     A[:, :, -(centre-1):, -(centre-1):] = a[:, :, :(centre-1), :(centre-1)]
#     return fft2(A)


# def FT2d(a_tensor):#since when we construct the OTF, the shift has been taken
#     if len(a_tensor.shape) == 4:
#         return fftn(a_tensor,dim =[2,3])
#     elif len(a_tensor.shape) == 3:
#         return fftn(a_tensor,dim =[1,2])

def batch_iFT2d(a_tensor):
    assert len(a_tensor.shape)==4, "expected dimension is 4 with batch size at first"
    return ifftshift(ifft2(fftshift(a_tensor,dim=[2,3])),dim= [2,3])

# def iFT2d(a_tensor):# since when we construct the OTF, the shift has been taken
#     if len(a_tensor.shape) == 4:
#         return ifftn(a_tensor,dim =[2,3])
#     elif len(a_tensor.shape) == 3:
#         return ifftn(a_tensor,dim =[1,2])

def generate_K1map(ob, block_shape, norm = False):
    Ks_m, Ks_n = block_shape
    Kt, ob_m, ob_n = ob.shape
    out_m, out_n = ob_m // Ks_m, ob_n // Ks_n
    out = np.zeros(shape=(out_m, out_n), dtype=np.float64)
    for i in range(out_m):
        for j in range(out_n):
            out[i][j] = np.sum(ob[:, i*Ks_m:(i+1)*Ks_m, j*Ks_n:(j+1)*Ks_n])
    if norm:
        return out/(Ks_m*Ks_n*Kt)
    return out

def blockfunc( ob, block_shape, func): # clear
    """
    Parameters:
        :ob: the observation of single photon imaging
        :block_shape: block shpae of this operation
        :func: the function that is applied to each block
    """

    # precompute some variables
    ob_m, ob_n = ob.shape

    # block shape
    b_m, b_n = block_shape

    # define the size of resulting image
    out_m, out_n = ob_m // b_m, ob_n // b_n

    # placeholder for the output
    out = np.zeros(shape=(out_m, out_n), dtype=np.float64)

    for i in range(out_m):
        for j in range(out_n):
            out[i][j] = func(ob[i*b_m:(i+1)*b_m, j*b_n:(j+1)*b_n])

    return out

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p



# %%-------------------------------------- Training metrics ------------------------------------------

def NPCC_loss(y_pred,y_true):
    [B,D,W,H] = y_pred.shape
    y_pred = y_pred.view(-1,D*W*H)
    y_true = y_true.view(-1,D*W*H)
    y_pred_m = torch.mean(y_pred, dim=1, keepdim=True)
    y_true_m = torch.mean(y_true, dim=1, keepdim=True)
    vp = (y_pred-y_pred_m)
    vt = (y_true-y_true_m)
    c = torch.mean(vp*vt, dim=1)/(torch.sqrt(torch.mean(vp**2, dim=1)+1e-08) * torch.sqrt(torch.mean(vt ** 2,dim=1)+1e-08))
    loss = torch.mean(1-c**2) # torch.mean(1-c**2)
    return loss

def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))

def accuracy(y_pred,y_true):
    """Computes the accuracy for multiple binary predictions"""
    pred = y_pred >= 0.5
    truth = y_true >= 0.5
    acc = pred.eq(truth).sum() / y_true.numel()
    return acc

def mse_loss(y_pred,y_true):
    return torch.nn.functional.mse_loss(y_pred,y_true)

def PSNR(y_pred, y_true):
    y_pred = norm_tensor(y_pred)
    y_true = norm_tensor(y_true)
    EPS = 1e-8
    mse = torch.mean((y_pred - y_true) ** 2,dim=[2,3])
    score  = -10*torch.log10(mse+EPS)
    score  = torch.mean(score)
    # score = - 10 * torch.log10(mse + EPS)
    return score

def PCC(y_pred,y_true, mean=True):
    [B,D,W,H]  = y_pred.shape
    y_pred = y_pred.view([B,D*W*H])
    y_true = y_true.view([B,D*W*H])
    mp = torch.mean(y_pred,dim =1,keepdim=True)
    mt = torch.mean(y_true,dim =1,keepdim=True)
    x_p,x_t = y_pred-mp,y_true-mt
    # std_p = torch.std(y_pred,dim=1)
    # std_t = torch.std(y_true,dim=1)

    num = torch.mean(torch.mul(x_p,x_t),dim=1)
    den = torch.norm(x_p,p=2,dim=1)*torch.norm(x_t,p=2,dim=1)
    # den = std_p*std_t
    if mean:
        return torch.mean(num/den)
    return num/den


def acc_and_recall_with_buffer(y_pred,label,buffer=10,mean=True,threshold=0.5,use_centroids=False,grouped=True):
    '''

    :param y_pred: [B,D,W,H] directly from the network => transmittance prediction 0 for particle, 1 for no particle
    :param label: [B,D,W, H] particle prediction => 0 for no particle, 1 for particle
    :param buffer: buffer range in pixel
    :param buffer: whether to use groupped label
    :param mean:
    :return:
    '''

    [B,D,W,H] = y_pred.shape
    recall = []
    acc = []
    for i in range(B):
        y_t = label[i,:,:,:]
        y_p = y_pred[i,:,:,:]
        idx = (y_p>=threshold)
        temp = np.zeros_like(y_p)
        temp[idx] = np.ones_like(y_p)[idx] # transmittance
        # [_recall, _acc]= prediction_metric(1-temp,y_t,buffer,threshold=threshold,use_centroids=use_centroids)# transmittance need to transfer to particle existance
        [_recall, _acc]= prediction_metric(1-temp,y_t,buffer,threshold=threshold,use_centroids=use_centroids,grouped=grouped)# transmittance need to transfer to particle existance
        recall.append(_recall)
        acc.append(_acc)
    if mean:
        return np.mean(recall),np.mean(acc)
    return recall,mean


# if __name__ == "__main__":
#     y_pred = torch.zeros([2,3,4,4])
#     a = 2*torch.ones([2,2])
# %%-------------------------------------- Debug helper ------------------------------------------
def inner_check_vis(x,phi,z,u1,u2,prefix,dir='./vis_for_check/'):
    '''

    :param x: [B,D, H,W]
    :param phi: [B,1,H,W]
    :param z: [B,D,H,W]
    :param u1: [B,1,H,W]
    :param u2: [B,D,H,W]
    :return:plot imgs
    '''
    import os
    if not os.path.isdir(dir):
        os.makedirs(dir)
    x = tensor2value(x)
    x = x[0,:,:,:]
    plotcube(x,'x_update',file_name=dir+prefix+'_x.png',show=False,cmap='Greys')
    phi = tensor2value(phi)
    phi = phi[0,0,:,:]
    plot_img(phi,'phi_update',fig_name=dir+prefix+'_phi.png',show=False,cmap='Greys')
    z = tensor2value(z)
    z = z[0,:,:,:]
    plotcube(z,'z_update',file_name=dir+prefix+'_z.png',show=False,cmap='Greys')


# %%-------------------------------------- Display ------------------------------------------
# Plot a 3D matrix slice by slice
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

def plot_img(img,fig_title,fig_name=None,show =False, **kwargs):
    if len(img.shape) == 4:
        img = img[1,0,:,:]
    elif len(img.shape)==3:
        img = img[0,:,:]
    plt.figure()
    plt.imshow(img, **kwargs)
    plt.title(fig_title)
    if show:
        plt.show()
    if fig_name:
        plt.savefig(fig_name)

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

# def eval_metric_withR(pred,label,buffer=10, threshold=0.5, use_centroid=False):
#     '''
#     :param pred: [D,H,W] the prediction for particle
#     :param label: [D,H,W] the label for particle
#     :param buffer:
#     :param threshold:
#     :param use_centroid:
#     :return:
#     '''
#     [D,H,W] = pred.shape
#     if type(pred) is not np.array:
#         pred = np.array(pred)
#     if type(label) is not np.array:
#         gt = np.array(label)
#
#     check = np.where(pred > threshold)[0]
#
#     # need to group gt,the returned gt is the single-pixel representation
#     _,gt_centers = group_gt(label,radius=5,n_p=5)
#     gt = np.zeros_like(label)
#     gt[gt_centers[:,0],gt_centers[:,1],gt_centers[:,2]] = np.ones_like(label)[gt_centers[:,0],gt_centers[:,1],gt_centers[:,2]]
#     mask = generate_buffer_field(gt,buffer)
#
#     if use_centroid:
#         _, pred_centers =  process_cube(pred,radius=10,n_p=10)
#         pred_p = np.zeros_like(pred)
#         pred_p[pred_centers[:,0],pred_centers[:,1],pred_centers[:,2]] = np.ones_like(pred)[pred_centers[:,0],pred_centers[:,1],pred_centers[:,2]]
#     else:
#         pred_p = pred
#
#     TP = mask * pred_p
#     FP = (1-mask)* pred_p
#     FN = np.sum(gt > threshold)-np.sum(TP) #missed particles
#     recall = np.sum(TP)/(np.sum(TP)+FN+1e-5)
#     if recall>=1:
#         recall=0
#     acc = np.sum(TP)/(np.sum(TP)+np.sum(FP)+1e-5)
#     if acc>=1:
#         acc=0
#     return recall,acc

def prediction_metric(predictions,label,buffer=10,threshold = 0.5, use_centroids=False,grouped=True):
    '''

    :param predictions: [D,H,W] the prediction with particle 1, no particle 0
    :param gt: [D,H,W] the gt with particle 1, no particle 0
    :param buffer: buffer range for gt
    :param threshold:
    :return:
    '''
    [D,H,W] = predictions.shape
    if type(predictions) is not np.array:
        predictions = np.array(predictions)
    if type(label) is not np.array:
        label = np.array(label)

    if use_centroids:
        gt = label
        _, pred_centers = process_cube(predictions,radius=10,n_p=10)
        pred = np.zeros_like(predictions)
        pred[pred_centers[:,0],pred_centers[:,1],pred_centers[:,2]] = np.ones_like(predictions)[pred_centers[:,0],pred_centers[:,1],pred_centers[:,2]]
    else:
        gt = label
        pred = predictions

    mask,gt = generate_buffer_field(gt,buffer,grouped=grouped,radius=10,n_p=5) #此处的gt是已经grouped过的,radius 和 n_p 要根据real particle来决定
    # mask = (field_buffer!=0)
    TP = mask * pred
    FP = (1-mask)* pred
    if use_centroids:
        FN = np.sum(gt > threshold)-np.sum(TP)# missed particles
    else:
        FN = np.sum(label > threshold)-np.sum(TP) # approximate missed particles

    recall = np.sum(TP)/(np.sum(TP)+FN+1e-5)
    if recall>=1:
        recall=0
    acc = np.sum(TP)/(np.sum(TP)+np.sum(FP)+1e-5)
    if acc>=1:
        acc=0
    return recall,acc

def process_cube(ori, **kwargs):
    [D,H,W] = ori.shape
    out = np.zeros([D,H,W])
    c = []
    for i in range(D):
        temp = ori[i,:,:]
        out_slice, ctr = get_filtered_centroids(temp,**kwargs)
        ctr_ext = np.concatenate([np.array([i]*len(ctr))[:,None],ctr],axis = -1)
        out[i,:,:] = out_slice
        c.extend(ctr_ext.tolist())
    return out, np.array(c)

def group_gt(gt, radius=10,n_p=5):
    [D,H,W] = gt.shape
    out = np.zeros([D,H,W])
    c = []
    for i in range(D):
        temp = gt[i,:,:]
        out_slice, ctr = get_filtered_centroids(temp,radius=radius,n_p=n_p)
        ctr_ext = np.concatenate([np.array([i]*len(ctr))[:,None],ctr],axis = -1)
        out[i,:,:] = out_slice
        c.extend(ctr_ext.tolist())
    centers = np.array(c)
    p_vol = np.zeros_like(gt)
    p_vol[centers[:,0],centers[:,1],centers[:,2]] = np.ones_like(gt)[centers[:,0],centers[:,1],centers[:,2]]
    return p_vol,centers

def filter_by_radius(img,radius=10, n_p = None):
    """
    :param img: shall be volume slice with predicted particle location (1 for particle exist, 0 for no particle)
    :param radius: range for searching the surrounding (unit: pixel)  gt里面为25个particle聚集为一个点
    :return: filtered slice
    """
    if not n_p:
        n_p = round((0.5*radius)**2*0.8)
    idx = np.where(img==1)
    idx = np.transpose(np.asarray(idx)) #[M,2]
    dist =np.sqrt(np.sum((idx[:,None,:]-idx)**2,axis=-1)) #distance matrix [M,M]
    candidate = np.zeros_like(dist)
    candidate[dist<radius] = np.ones_like(dist)[dist<radius]
    p_num = np.sum(candidate,axis=1)
    mask = (p_num>=n_p)
    idx_filtered = idx[mask,:]
    img_filetered = np.zeros_like(img)
    img_filetered[idx_filtered[:,0],idx_filtered[:,1]] = np.ones_like(img)[idx_filtered[:,0],idx_filtered[:,1]]
    return img_filetered

def get_filtered_centroids(img,radius=10,n_p =None):
    """
    :param img: shall be volume slice with predicted particle location (1 for particle exist, 0 for no particle)
    :param radius: range for searching the surrounding (unit: pixel)  gt里面为25个particle聚集为一个点
    :return: img_filetered: filtered slice
             centers_filtered: [z,x,y] for corresponding filtered centers
    """
    assert len(img.shape)==2, "Received shape: %d should be 2D image not 3D block" %(img.shape)
    if not n_p:
        n_p = np.round((0.5*radius)**2*0.8)
    idx = np.where(img==1)
    idx = np.transpose(np.asarray(idx)) #[M,2]

    if idx.shape[0]>2000:
        return img,np.zeros([2,2]).astype(np.int)
    dist =np.sqrt(np.sum((idx[:,None,:]-idx)**2,axis=-1)) #distance matrix [M,M]

    candidate = np.zeros_like(dist)
    candidate[dist<radius] = np.ones_like(dist)[dist<radius]
    p_num = np.sum(candidate,axis=1)
    mask = (p_num>=n_p)

    centers = np.dot(candidate,idx)/p_num[:,None]
    centers_filtered = centers[mask,:]
    centers_filtered = np.round(centers_filtered).astype(np.int)
    if len(centers_filtered)!=0:
        centers_filtered = np.unique(centers_filtered,axis=0)

    idx_filtered = idx[mask,:]
    img_filtered = np.zeros_like(img)
    img_filtered[idx_filtered[:,0],idx_filtered[:,1]] = np.ones_like(img)[idx_filtered[:,0],idx_filtered[:,1]]

    return img_filtered, centers_filtered

def distance_array(vec1,vec2):
    '''
    :param vec1: array [N,dim]
    :param vec2: array [M,dim]
    :return: distance matrix [N,M]
    '''
    out =np.sqrt(np.sum((vec1[:,None,:]-vec2)**2,axis=-1))
    return out

def write_to_mat(tensor, pred=False,gt=False,prefix = None,channel_first=True):
    '''

    :param pred: [B,D,H,W]
    :param gt:  [B,D,H,W]
    :param prefix:
    :return:
    '''
    if channel_first:
        if pred:
            pred = tensor.permute([0,2,3,1])
            scipy.io.savemat( prefix+'predict.mat',{'predict':tensor2value(pred)})
        if gt:
            gt = tensor.permute([0,2,3,1])
            scipy.io.savemat( prefix+'gt.mat',{'gt':tensor2value(gt)})
    else:
        if pred:
            scipy.io.savemat( prefix+'predict.mat',{'predict':tensor2value(tensor)})
        if gt:
            scipy.io.savemat( prefix+'gt.mat',{'gt':tensor2value(tensor)})


def filter_out_margin(cube, margin=[0,5,5]):
    [z,x,y] = cube.shape
    [bz,bx,by] = margin
    out = np.zeros_like(cube)
    for l in range(bz,z-bz):
        out[l,bx:(x-bx),by:(y-by)] = cube[l,bx:(x-bx),by:(y-by)]
    return out

def generate_otf_torch(wavelength, nx, ny, deltax, deltay, distance, pad_size=None):
    """
    Generate the otf from [0,pi] not [-pi/2,pi/2] using torch
    :param wavelength:
    :param nx:
    :param ny:
    :param deltax:
    :param deltay:
    :param distance:
    :return:
    """
    if pad_size:
        nx = pad_size[0]
        ny = pad_size[1]
    r1 = torch.linspace(-nx / 2, nx / 2 - 1, nx)
    c1 = torch.linspace(-ny / 2, ny / 2 - 1, ny)
    deltaFx = 1 / (nx * deltax) * r1
    deltaFy = 1 / (nx * deltay) * c1
    mesh_qx, mesh_qy = torch.meshgrid(deltaFx, deltaFy)
    k = 2 * torch.pi / wavelength
    otf = np.exp(1j * k * distance * torch.sqrt(1 - wavelength ** 2 * (mesh_qx ** 2
                                                                       + mesh_qy ** 2)))
    otf = torch.fft.ifftshift(otf)
    return otf



if __name__== "__main__":
    gt = torch.zeros([18,7,512,512]).type(torch.float32)
    gt[0,0,0,2] = 1
    gt[0,1,2,1] = 1
    gt[0,2,2,2] = 1
    pred = torch.ones([18,7,512,512]).type(torch.float32)
    pred[0,0,0,2] = 0
    pred[0,1,0,0] = 0
    [recall,acc]=acc_and_recall_with_buffer(pred,gt,buffer=10,mean=True,threshold=0.5)
