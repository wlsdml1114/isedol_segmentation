
import os
import cv2
from tqdm import tqdm
#from demo.image_matting.colab.inference2 import run
import numpy as np
from collections import Counter
import argparse
import torchvision.transforms.functional as TF
import configparser
from PIL import Image

config = configparser.ConfigParser()    
config.read('./setting.ini', encoding='CP949') 

file_names = os.listdir(config['path']['ori_dir'])
tr_data_dir = config['path']['tr_data_dir']
aug_dir = config['path']['aug_dir']

for folder in file_names:
    print()
    '''file_name = file_name.split('/')
    folder = file_name[-1]'''   
    print(folder,'data augmentation start')

    files = os.listdir(os.path.join(tr_data_dir,folder,"Masks"))
    if not(os.path.exists(os.path.join(aug_dir,folder,"Images"))):
        os.system('mkdir -p '+os.path.join(aug_dir,folder,"Images"))

    if not(os.path.exists(os.path.join(aug_dir,folder,"Masks"))):
        os.system('mkdir -p '+os.path.join(aug_dir,folder,"Masks"))

    for idx in tqdm(range(len(files))):
        img = cv2.imread(os.path.join(tr_data_dir,folder,"Images",files[idx][:-3]+'jpg'))      
        try :  
            pil_image = Image.fromarray(img)
        except :
            print(folder,files[idx],"Exception occur")
            continue
        flip_img = TF.hflip(pil_image)
        bright_img=TF.adjust_brightness(pil_image,0.7)
        cont_img=TF.adjust_contrast(pil_image,0.8)
        crop_img = img[100:,100:]
        cv2.imwrite(os.path.join(aug_dir,folder,"Images",'ori_%sjpg'%(files[idx][:-3])),img)
        cv2.imwrite(os.path.join(aug_dir,folder,"Images",'flip_%sjpg'%(files[idx][:-3])),np.array(flip_img))
        cv2.imwrite(os.path.join(aug_dir,folder,"Images",'bri_%sjpg'%(files[idx][:-3])),np.array(bright_img))
        cv2.imwrite(os.path.join(aug_dir,folder,"Images",'cont_%sjpg'%(files[idx][:-3])),np.array(cont_img))
        cv2.imwrite(os.path.join(aug_dir,folder,"Images",'crop_%sjpg'%(files[idx][:-3])),crop_img)



        img = cv2.imread(os.path.join(tr_data_dir,folder,"Masks",files[idx]))
        pil_image = Image.fromarray(img)
        flip_img = TF.hflip(pil_image)
        bright_img=TF.adjust_brightness(pil_image,0.7)
        cont_img=TF.adjust_contrast(pil_image,0.8)
        crop_img = img[100:,100:]
        cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'ori_%spng'%(files[idx][:-3])),img)
        cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'flip_%spng'%(files[idx][:-3])),np.array(flip_img))
        cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'bri_%spng'%(files[idx][:-3])),np.array(bright_img))
        cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'cont_%spng'%(files[idx][:-3])),np.array(cont_img))
        cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'crop_%spng'%(files[idx][:-3])),crop_img)

