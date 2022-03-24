
import os
import cv2
from tqdm import tqdm
#from demo.image_matting.colab.inference2 import run
import numpy as np
from collections import Counter
import argparse
import configparser

config = configparser.ConfigParser()    
config.read('./setting.ini', encoding='CP949') 

file_names = config['path']['name'].split('/')
folder = file_names[-1]
tr_data_dir = config['path']['tr_data_dir']
aug_dir = config['path']['aug_dir']

files = os.listdir(os.path.join(tr_data_dir,folder,"Masks"))
if not(os.path.exists(os.path.join(aug_dir,folder,"Images"))):
    os.system('mkdir -p '+os.path.join(aug_dir,folder,"Images"))

if not(os.path.exists(os.path.join(aug_dir,folder,"Masks"))):
    os.system('mkdir -p '+os.path.join(aug_dir,folder,"Masks"))

print('mask_counting')
pix = []
for idx in tqdm(range(len(files))):
    img = cv2.imread(os.path.join(tr_data_dir,folder,"Images",files[idx][:-3]+'jpg'))
    mask = cv2.imread(os.path.join(tr_data_dir,folder,"Masks",files[idx]))

