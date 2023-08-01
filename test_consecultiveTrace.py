'''
This manuscript is used to test the consecutive trace that is simulated from recorded particle experimental trace
the data is stored as 'data/ConsecutiveTrace' which is generated from the 'data/ConsecutiveTrace/OriginalDataLabel'
Can be directly compiled using CPU
'''
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import torch
import logging
from model.PLholonet import PLholonet
from utils.dataset import create_dataloader_Poisson
from utils.utilis import PCC, PSNR, accuracy, random_init, tensor2value, plotcube, acc_and_recall_with_buffer, \
    prediction_metric,generate_buffer_field
import matplotlib.pyplot as plt
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value,plotcube,process_cube,get_filtered_centroids
from utils.evaluation_utils import compare_cube
from scipy.io import savemat,loadmat
# %%---------------------Parse the consecutive label from .txt into .mat then generate data using Matlab script----------------

from pathlib import Path
from scipy.io import savemat

txt_dir = "data/ConsecutiveTrace/OriginalDataLabel"
f_n = [str(x) for x in Path(txt_dir).resolve().glob('*.txt')]
f_n.sort(key = lambda x: int(x.split('/')[-1].split('.')[0]))
def parse_txt_label(txt_fp='.txt',Nxy = 256,n_layer_crt=7,n_layer_ori=256):
    with open(Path(txt_fp).with_suffix('.txt'), 'r') as file:
        l = [x.split() for x in file.read().strip().splitlines() if len(x)]
        l = np.array(l, dtype=np.float32)
        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
        x_c = (l[:,1]*Nxy).astype(int)
        y_c = (l[:,2]*Nxy).astype(int)
        z_c = (l[:,0]/n_layer_ori*n_layer_crt).astype(int)
        return np.array([z_c,y_c,x_c]).transpose()


for f in f_n:
    txt_label = parse_txt_label(f)
    prefix = f.split('/')[-1].split('.')[0]
    savemat('data/ConsecutiveTrace/LabelinMat'+prefix+'.mat', {'arr': txt_label})


# %%---------------------Evaluate the consecutive trace data and save output ----------------
def cal_acc_recall(gt,pred,buffer = 20,field= [7,256,256], threshold = 0.5):
    gt = gt.astype(np.int)
    pred = pred.astype(np.int)
    gt_field = np.zeros(field)
    gt_field[gt[:,0],gt[:,1],gt[:,2]] = np.ones_like(gt_field)[gt[:,0],gt[:,1],gt[:,2]]
    pred_field = np.zeros(field)
    pred_field[pred[:,0],pred[:,1],pred[:,2]] = np.ones_like(pred_field)[pred[:,0],pred[:,1],pred[:,2]]
    mask,gt_field = generate_buffer_field(gt_field,buffer,grouped=False,radius=10,n_p=5)
    TP = mask * pred_field
    FP = (1-mask)* pred_field
    FN = np.sum(gt_field > threshold)-np.sum(TP)# missed particles
    recall = np.sum(TP)/(np.sum(TP)+FN+1e-5)
    if recall>=1:
        recall=0
    acc = np.sum(TP)/(np.sum(TP)+np.sum(FP)+1e-5)
    if acc>=1:
        acc=0
    return recall,acc

def norm_tensor(x):
    return (x-torch.min(x))/(torch.max(x)-torch.min(x))
def psnr(x,im_orig):
    x = norm_tensor(x)
    im_orig = norm_tensor(im_orig)
    mse = torch.mean(torch.square(im_orig - x))
    psnr = torch.tensor(10.0)* torch.log10(1/ mse)
    return psnr

def render_2d(pred_ctr,gt_ctr,sz=25,label=['pred','gt'],edgecolors=['g','r']):
    fig = plt.figure()
    ax =fig.add_subplot(111)
    ax.scatter(pred_ctr[:,2],pred_ctr[:,1], c='white',edgecolors=edgecolors[0],marker='o',s =sz,label=label[0])
    ax.scatter(gt_ctr[:,2],gt_ctr[:,1], c='white',edgecolors=edgecolors[1],marker='^',s =sz,label=label[1])
    ax.set_title('xy projection')

    fig2 = plt.figure()
    ax2 =fig2.add_subplot(111)
    ax2.scatter(pred_ctr[:,0],pred_ctr[:,1], c="white",edgecolors=edgecolors[0],marker='o',s =sz,label=label[0])
    ax2.scatter(gt_ctr[:,0],gt_ctr[:,1], c='white',edgecolors=edgecolors[1],marker='^',s =sz,label=label[1])
    ax2.set_title('yz projection')

    return ax,ax2
# %%--------------------------------------Load different model under specific noise level-------------------------------
ALPHA = 20
data_path = "data/ConsecutiveTrace/Nxy256_Nz7_dz6.9mm_pps13.8um_lambda660nm"
# model_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/experiment/PLHolo_train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm_L5_B32_lr0.001_Gamma1/Alpha100/last.pt"
# model_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/experiment/PLHolo_train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm_L5_B32_lr0.001_Gamma1/Alpha40/best.pt"
model_path = "/Users/zhangyunping/PycharmProjects/PLParticleTracking/experiment/PLHolo_train_Nxy256_Nz7_ppv1.1e-04_dz6.9mm_pps13.8um_lambda660nm_L5_B32_lr0.001_Gamma1_Alpha20/2023-03-21-10_59_33/best.pt"
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
model.eval()

for i,(y, label, otf3d) in enumerate(dataloader):

    if i in [0,1,2,12,23,34,45,56,67,78,89]: # this is for some non-consecutive frames clean out
        continue
    with torch.no_grad():
        y = y.to(device=model.device)
        otf3d = otf3d.to(torch.complex64).to(device=model.device)
        label = label.to(torch.float32).to(device=model.device)
        # t1=time.time()
        x, _sloss = model(y, otf3d)
        # t2=time.time()
        # print(t2-t1)
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
        # %%--------------------------------------plot the prediction and GT in 2D------------------------------------------
        # plotcube(gt, 'GT', "Gt.png", show=True)
        # plotcube(pred_cube, 'P_A%.3f_R%.3f' % (acc,recall), "pred.png", show=True)

        # %%--------------------------------------only plot the prediction in 3D------------------------------------------
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        # ax = compare_cube(gt,gt,15,ax=ax,sz=20)
        ax = compare_cube(1 - pred_cube,1 - pred_cube,15,ax=ax,sz=40)
        ax.set_xlim3d(0,256)
        ax.set_ylim3d(0,256)
        ax.set_zlim3d(0,6)
        plt.savefig('output/pred_{:d}.png'.format(i))
        # %%--------------------------------------only plot the noisy holograms------------------------------------------
        fig = plt.figure()
        plt.imshow(y[0, 0, :, :].tolist(), cmap='gray')
        plt.axis(False)
        plt.savefig('output/holo_{:d}.png'.format(i))
        print(i)



    # ax.tick_params(axis='z', which='major', labelsize=17)
    # ax.set_title('Precision: %.3f Recall:%.3f'%(recall,acc))
    # ax.legend(["correct","wrong","missed"],loc = 'upper center',ncol= 3,fontsize=10)
    # plt.show()

    # render_2d(pred_cts,gt_cts,label=['pred','GT'],edgecolors=['r','g'])
    # render_2d(gt_cts,pred_cts,label=['GT','pred'])
