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

ori_dir = '/home/jini1114/git/data/dataset'
aug_dir = '/home/jini1114/git/data/augmentation'

folders = os.listdir(ori_dir)


for folder in folders:

    print(folder,'data augmentation start')

    for num in os.listdir(os.path.join(ori_dir,folder)):
        if 'DS_Store' in num:
            continue
        files = os.listdir(os.path.join(ori_dir,folder,num,"Masks"))
        if not(os.path.exists(os.path.join(aug_dir,folder,"Images"))):
            os.system('mkdir -p '+os.path.join(aug_dir,folder,"Images"))

        if not(os.path.exists(os.path.join(aug_dir,folder,"Masks"))):
            os.system('mkdir -p '+os.path.join(aug_dir,folder,"Masks"))

        for idx in tqdm(range(len(files))):
            img = cv2.imread(os.path.join(ori_dir,folder,num,"Images",files[idx][:-3]+'jpg'))      
            try :  
                pil_image = Image.fromarray(img)
            except :
                print(folder,files[idx],"Exception occur")
                continue
            flip_img = TF.hflip(pil_image)
            bright_img=TF.adjust_brightness(pil_image,0.7)
            cont_img=TF.adjust_contrast(pil_image,0.8)
            crop_img = img[100:,100:]
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'ori_%s_%sjpg'%(num,files[idx][:-3])),img)
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'flip_%s_%sjpg'%(num,files[idx][:-3])),np.array(flip_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'bri_%s_%sjpg'%(num,files[idx][:-3])),np.array(bright_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'cont_%s_%sjpg'%(num,files[idx][:-3])),np.array(cont_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'crop_%s_%sjpg'%(num,files[idx][:-3])),crop_img)



            img = cv2.imread(os.path.join(ori_dir,folder,num,"Masks",files[idx]))
            pil_image = Image.fromarray(img)
            flip_img = TF.hflip(pil_image)
            bright_img=TF.adjust_brightness(pil_image,0.7)
            cont_img=TF.adjust_contrast(pil_image,0.8)
            crop_img = img[100:,100:]
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'ori_%s_%spng'%(num,files[idx][:-3])),img)
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'flip_%s_%spng'%(num,files[idx][:-3])),np.array(flip_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'bri_%s_%spng'%(num,files[idx][:-3])),np.array(bright_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'cont_%s_%spng'%(num,files[idx][:-3])),np.array(cont_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'crop_%s_%spng'%(num,files[idx][:-3])),crop_img)

'''
config = configparser.ConfigParser()    
config.read('./setting.ini', encoding='CP949') 

ori_dir = '/data/codes/data/dataset'
aug_dir = '/data/codes/data/augmen'

folders = os.listdir(ori_dir)


for folder in folders:

    print(folder,'data augmentation start')

    for num in os.listdir(os.path.join(ori_dir,folder)):
        if 'DS_Store' in num:
            continue
        files = os.listdir(os.path.join(ori_dir,folder,num,"Masks"))
        if not(os.path.exists(os.path.join(aug_dir,folder,"Images"))):
            os.system('mkdir -p '+os.path.join(aug_dir,folder,"Images"))

        if not(os.path.exists(os.path.join(aug_dir,folder,"Masks"))):
            os.system('mkdir -p '+os.path.join(aug_dir,folder,"Masks"))

        for idx in tqdm(range(len(files))):
            img = cv2.imread(os.path.join(ori_dir,folder,num,"Images",files[idx][:-3]+'jpg'))      
            try :  
                pil_image = Image.fromarray(img)
            except :
                print(folder,files[idx],"Exception occur")
                continue
            flip_img = TF.hflip(pil_image)
            bright_img=TF.adjust_brightness(pil_image,0.7)
            cont_img=TF.adjust_contrast(pil_image,0.8)
            crop_img = img[100:,100:]
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'ori_%s_%sjpg'%(num,files[idx][:-3])),img)
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'flip_%s_%sjpg'%(num,files[idx][:-3])),np.array(flip_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'bri_%s_%sjpg'%(num,files[idx][:-3])),np.array(bright_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'cont_%s_%sjpg'%(num,files[idx][:-3])),np.array(cont_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Images",'crop_%s_%sjpg'%(num,files[idx][:-3])),crop_img)



            img = cv2.imread(os.path.join(ori_dir,folder,num,"Masks",files[idx]))
            pil_image = Image.fromarray(img)
            flip_img = TF.hflip(pil_image)
            bright_img=TF.adjust_brightness(pil_image,0.7)
            cont_img=TF.adjust_contrast(pil_image,0.8)
            crop_img = img[100:,100:]
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'ori_%s_%spng'%(num,files[idx][:-3])),img)
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'flip_%s_%spng'%(num,files[idx][:-3])),np.array(flip_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'bri_%s_%spng'%(num,files[idx][:-3])),np.array(bright_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'cont_%s_%spng'%(num,files[idx][:-3])),np.array(cont_img))
            cv2.imwrite(os.path.join(aug_dir,folder,"Masks",'crop_%s_%spng'%(num,files[idx][:-3])),crop_img)

'''