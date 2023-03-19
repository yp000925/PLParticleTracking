from utils.evaluation_utils import compare_cube
import matplotlib.pyplot as plt

import numpy as np
import os
import time
import torch
import logging
from model.PLholonet import PLholonet
from utils.dataset import create_dataloader_qis
from utils.utilis import PCC,PSNR,accuracy,random_init,tensor2value,plotcube,prediction_metric,process_cube,get_filtered_centroids,filter_out_margin

import re
def numerical_eval(dataloader,img_num,model):
    model.eval()
    with torch.no_grad():
        for i,(K1_map, K0_map, label, otf3d, y) in enumerate(dataloader):

            K1_map = K1_map.to(torch.float32).to(device=model.device)
            otf3d = otf3d.to(torch.complex64).to(device=model.device)
            label = label.to(torch.float32).to(device=model.device)
            x, _sloss = model(K1_map,K0_map,otf3d)
            gt = torch.ones_like(label)-label
            # _dloss = torch.mean(torch.pow(x-gt,2))
            _dloss = torch.nn.BCELoss()(x, gt)
            if len(x.shape)==4:
                x = x.squeeze(0)

            x= tensor2value(x)
            gt = tensor2value(label.squeeze(0))
            threds = np.arange(start=0.3,stop=0.6,step=0.02)
            out = np.zeros([img_num,len(threds),3])
            if i >= img_num:
                break
            else:
                for j in range(len(threds)) :
                    pred_cube = np.zeros_like(x)
                    idx = (x>=threds[j])# threshold value lower => predicted particle sparser
                    pred_cube[idx] =  np.ones_like(x)[idx]
                    prediction = 1-pred_cube
                    [recall,precision] = prediction_metric(prediction,gt,15)
                    out[i,j,0] = threds[j]
                    out[i,j,1] = recall
                    out[i,j,2] = precision
            return out

if __name__ == "__main__":
    # load the test dataset
    batch_sz = 1
    # model_path = "/Users/zhangyunping/PycharmProjects/PLParticle/experiment/PLHolo_train_Nz7_Nxy32_kt100_ks16_ppv1e-03~5e-03_pps2e-05_z05e-03_dz1e-03_L5_B32_lr0.0001_Gamma0.1/last.pt"
    model_path = "/Users/zhangyunping/PycharmProjects/PLParticle/experiment/PLHolo_train_Nz30_Nxy128_kt100_ks8_ppv5e-04~1e-03_pps2e-05_z05e-03_dz1e-03_L5_B32_lr0.0001_Gamma0.001/1020.pt"
    data_file_base = "/Users/zhangyunping/PycharmProjects/PLParticle/data/Kmap_Particle/"
    # val_data_name = "val_Nz7_Nxy32_kt100_ks16_ppv1e-03~5e-03_pps2e-05_z05e-03_dz1e-03"
    val_data_name = "val_Nz30_Nxy128_kt100_ks8_ppv5e-04~1e-03_pps2e-05_z05e-03_dz1e-03"
    val_data_path = data_file_base + val_data_name
    params = val_data_name.split('_')
    kt = [eval(re.findall(r'kt(\d+)',x)[0]) for x in params if re.findall(r'kt(\d+)',x)][0]
    ks = [eval(re.findall(r'ks(\d+)',x)[0]) for x in params if re.findall(r'ks(\d+)',x)][0]

    data_loader,dataset = create_dataloader_qis(val_data_path,batch_sz,kt,ks,norm=True,lite=True)
    data = iter(data_loader)


    #load the model
    model_name = model_path.split('/')[-2]
    params = model_name.split('_')

    Nd = [eval(re.findall(r'L(\d+)',x)[0]) for x in params if re.findall(r'L(\d+)',x)][0]
    Nz = [eval(re.findall(r'Nz(\d+)',x)[0]) for x in params if re.findall(r'Nz(\d+)',x)][0]


    model = PLholonet(n=Nd, d=Nz)

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


    out_dir ="/Users/zhangyunping/PycharmProjects/PLParticle/Output_example/"
    prefix = model_name.split('/')[-1]
    out_dir = out_dir+prefix

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with torch.no_grad():
        # threds = np.arange(start=0.35,stop=0.5,step=0.02)
        threds = [0.45]
        ndata = 1
        out = np.zeros([ndata,len(threds),3])
        out2 = np.zeros([ndata,len(threds),3])
        picked_idx = 7
        for img_number in range(ndata):
            K1_map, K0_map, label, otf3d = data.next()
            # if img_number != picked_idx:
            #     continue
            K1_map = K1_map.to(torch.float32).to(device=model.device)
            otf3d = otf3d.to(torch.complex64).to(device=model.device)
            label = label.to(torch.float32).to(device=model.device)
            x, _sloss = model(K1_map,K0_map,otf3d)
            gt = torch.ones_like(label)-label
            # _dloss = torch.mean(torch.pow(x-gt,2))
            _dloss = torch.nn.BCELoss()(x, gt)
            _total_loss = _dloss + _sloss

            _pcc = PCC(x, label)
            _psnr = PSNR(x, label)
            _acc = accuracy(x, label)

            print(('\n' + '%10s' * 7) % ('  ', 'sloss', 'dloss', 'loss', 'acc', 'pcc', 'psnr'))
            info = ('%10s' + '%10.4g' * 6) % ('Test_result',_sloss, _dloss,
                                              _total_loss, _acc , _pcc, _psnr)
            print(info)

            x = tensor2value(x.squeeze(0))
            gt = tensor2value(label.squeeze(0))
            R = []
            A = []
            # for thred in threds:
            #     pred_cube = torch.zeros_like(x)
            #     idx = (x>=thred)# threshold value lower => predicted particle sparser
            #     pred_cube[idx] =  torch.ones_like(x)[idx]
            #     pred_cube = tensor2value(pred_cube)
            #     # plotcube(pred_cube,'predicted')
            #
            #     gt = tensor2value(label.squeeze(0))
            #     # plotcube(gt,'gt')
            #     prediction = 1-pred_cube
            #     [recall,acc] = prediction_metric(prediction,gt,buffer=10)
            #     print("Thred: %4f Recall: %4f Accuracy:%4f"%(thred,recall,acc))



            for j in range(len(threds)):
                temp = np.zeros_like(x)
                idx = (x>=threds[j])# threshold value lower => predicted particle sparser
                temp[idx] =  np.ones_like(x)[idx]
                pred_cube = 1-temp
                # plotcube(pred_cube,'ini_pred_thred_%.3f'%threds[j],out_dir+'/'+'ini_pred_thred_%.3f'%threds[j]+'.png',show=False)
                plotcube(pred_cube,'ini_pred_thred_%.3f'%threds[j])
                pred_cube_f = filter_out_margin(pred_cube,[0,10,10])
                gt_f = filter_out_margin(gt,[0,10,10])
                plotcube(pred_cube_f,'filtered_pred_thred_%.3f'%threds[j])
                # pred_cube_f,cts = process_cube(pred_cube,radius=5,n_p =5)
                # plotcube(pred_cube_f,'filted_pred_thred_%.3f'%threds[j],out_dir+'/'+'filted_pred_thred_%.3f'%threds[j]+'.png',show=False)
                # plotcube(pred_cube,'filted_pred_thred_%.3f'%threds[j])
                # [recall,precision] = prediction_metric(pred_cube_f,gt,5) # 之前是15
                [recall,precision] = prediction_metric(pred_cube,gt,buffer=10,grouped=False)
                [recall2,precision2] = prediction_metric(pred_cube_f,gt_f,buffer=10,grouped=False)
                out[img_number,j,0] = threds[j]
                out[img_number,j,1] = recall
                out[img_number,j,2] = precision
                out2[img_number,j,0] = threds[j]
                out2[img_number,j,1] = recall2
                out2[img_number,j,2] = precision2

                # fig = plt.figure()
                # ax = fig.add_subplot(111,projection = '3d')
                # ax = compare_cube(pred_cube_f,gt,15,ax=ax,sz=35)
                # ax.tick_params(axis='both', which='major', labelsize=14)
                # ax.set_title('Precision: %.3f Recall:%.3f'%(recall,precision))
                # ax.legend(["TP","FP","FN"])
                # temp = dataset.mat_files[img_number].split('/')[-1]
                # temp = temp.split('.')[0]
                # file_name = temp+'_thred_'+"%.3f"%threds[j] +'.png'
                # plt.savefig(out_dir+'/'+file_name)
                # # plt.show()
                # plt.close()
        pred_avg = np.mean(out,axis=0)

        print(pred_avg)
        pred_avg2 = np.mean(out2,axis=0)

        print(pred_avg2)




