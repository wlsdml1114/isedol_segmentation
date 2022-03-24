from configparser import Interpolation
import os
import cv2
from tqdm import tqdm
#from demo.image_matting.colab.inference2 import run
import numpy as np
from collections import Counter
import argparse


parser = argparse.ArgumentParser(description='mask rcnn')    
parser.add_argument('--name', required = True, help='folder name')
parser.add_argument('--jpg_dir', required = True, help='jpg dir path')
parser.add_argument('--png_dir', required = True, help='png dir path')
parser.add_argument('--tr_data_dir', required = True, help='tr data dir path')
args = parser.parse_args()
 
file_names = args.name.split('/')
folder = file_names[-1]
jpg_dir = args.jpg_dir
png_dir = args.png_dir
tr_data_dir = args.tr_data_dir

files = os.listdir(os.path.join(jpg_dir,folder))
if not(os.path.exists(os.path.join(tr_data_dir,folder,"Images"))):
    os.system('mkdir -p '+os.path.join(tr_data_dir,folder,"Images"))

if not(os.path.exists(os.path.join(tr_data_dir,folder,"Masks"))):
    os.system('mkdir -p '+os.path.join(tr_data_dir,folder,"Masks"))

print('mask_counting')
pix = []
for idx in tqdm(range(len(files))):
    mask = cv2.imread(os.path.join(png_dir,folder,files[idx][:-3]+'png'))
    mask[mask > 0.5] = 1
    mask[mask<=0.5] = 0
    pix.append(Counter(mask.flatten())[1])

cut = np.percentile(np.array(pix),[70,90])
lowcut = cut[0]
highcut = cut[1]

for idx in tqdm(range(len(files))):
    if (pix[idx] > lowcut) & (pix[idx]<highcut):
        os.system('cp '+os.path.join(jpg_dir,folder,files[idx])+' '+os.path.join(tr_data_dir,folder,'Images',files[idx]))
        os.system('cp '+os.path.join(png_dir,folder,files[idx][:-3]+'png')+' '+os.path.join(tr_data_dir,folder,'Masks',files[idx][:-3]+'png'))
