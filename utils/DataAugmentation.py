from PIL import Image,ImageEnhance,ImageOps,ImageFile,ImageChops
import numpy as np
import random
import threading,os,time
import logging
import math
import shutil

ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataAugmentation:
    def __init__(self):
        pass

    @staticmethod
    def open_Image(image):
        return Image.open(image).convert('L')

    @staticmethod
    def randomFlip(image):
        random_mode = np.random.randint(0,2)
        flip_mode = [Image.FLIP_LEFT_RIGHT,Image.FLIP_TOP_BOTTOM]
        return image.transpose(flip_mode[random_mode])

    @staticmethod
    def randomShift(image):
        random_xoffset = np.random.randint(0,math.ceil(image.size[0]*0.4))
        random_yoffset = np.random.randint(0,math.ceil(image.size[1]*0.4))
        return ImageChops.offset(image,xoffset = random_xoffset,yoffset = random_yoffset)

    @staticmethod
    def randomRotation(image,fillcolor = 255):
        deg = [90,180,270]
        random_angle = deg[np.random.randint(0,2)]
        return image.rotate(random_angle,fillcolor = fillcolor)

    @staticmethod
    def randomCrop(image,crop_w,crop_h):
        W,H = image.size
        x = random.randint(0,np.max([W-crop_w,0]))
        y = random.randint(0,np.max([H-crop_h,0]))
        random_region = (x,y,np.min([x+crop_w,W]),np.min([y+crop_h,H]))
        return image.crop(random_region).resize([crop_w,crop_h])

    @staticmethod
    def saveImage(image,path,**kwargs):
        try:
            image=image.convert('L')
            image.save(path,**kwargs)
        except:
            print('not save img:',path)
            pass

img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
files = []
import glob
from pathlib import Path
def get_files(dir_path):
    global files
    p = Path(dir_path)
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '**/*.*'),recursive=True))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    images = [x.replace('/',os.sep) for x in files if x.split('.')[-1].lower() in img_formats]
    return images

if __name__ == "__main__":
    img_dir = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/data/naturaldatasets"
    dst_dir = "/Users/zhangyunping/PycharmProjects/PLholo/syn_data/naturaldatasetsAug256"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    crop_w = 256
    crop_h = 256

    funcMap = {"flip":DataAugmentation.randomFlip,
               "rotation":DataAugmentation.randomRotation,
               "crop":DataAugmentation.randomCrop,
               "shift":DataAugmentation.randomShift
               }
    funcList = ["flip","rotation","crop"]

    imgs_list =get_files(img_dir)
    sample_p_img = 5
    for idx, img in enumerate(imgs_list):

        if idx !=0 and idx % 50 ==0:
            print("Now loaded %d images"%(idx))

        tmp_name = img.split('/')[-1]
        img_name = tmp_name.split('.')[0]
        img = DataAugmentation.open_Image(img)
        ori_img = img.resize([crop_w,crop_h])
        DataAugmentation.saveImage(ori_img,os.path.join(dst_dir,tmp_name))
        for i in range(sample_p_img):
            op_idx = random.randint(0,len(funcList)-1)
            func = funcList[op_idx]
            if func == 'crop':
                new_img = DataAugmentation.randomCrop(img,crop_w,crop_h)
            else:
                new_img = DataAugmentation.randomCrop(img,crop_w,crop_h)
                new_img = funcMap[func](new_img)
            img_path = os.path.join(dst_dir,img_name+'_'+str(i)+'.png')
            DataAugmentation.saveImage(new_img,img_path)

