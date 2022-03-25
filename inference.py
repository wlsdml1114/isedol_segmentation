import os
import numpy as np
import torch
import math
import cv2
import sys
import time
import argparse
from PIL import Image
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from typing import List, Tuple, Dict, Optional
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='mask rcnn')    
parser.add_argument('--file_name', required = True, help='folder name')
parser.add_argument('--model_dir', required = True, help='folder name')
parser.add_argument('--jpg_dir', required = True, help='folder name')
parser.add_argument('--seg_dir', required = True, help='folder name')
args = parser.parse_args()

names = args.file_name.split('/')
folder = names[-1]
seg_dir=args.seg_dir
jpg_dir=args.jpg_dir
model_dir = args.model_dir

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if not(os.path.exists(os.path.join(seg_dir,folder))):
    os.system('mkdir -p '+os.path.join(seg_dir,folder))
    
model_name = '%s.pt'%(folder[6:-4]) 
        
model = torch.load(os.path.join(model_dir,model_name))
model.eval()
model.to(device)

files = os.listdir(os.path.join(jpg_dir,folder))

for idx in tqdm(range(len(files))):
    img = cv2.imread(os.path.join(jpg_dir,folder,'%d.jpg'%(idx)))
    img = img.astype(np.float32)
    img = img/255

    output = model(torch.tensor([img.transpose(2,0,1)]).to(device))
    cv2.imwrite(os.path.join(seg_dir,folder,'%d.png'%(idx)),
                output[0]['masks'].detach().cpu().numpy()[0][0])
