import os
import numpy as np
import torch
import cv2
import argparse
from PIL import Image
from datetime import datetime, timedelta
from tqdm import tqdm
import datetime
import requests
print(datetime.datetime.now())

parser = argparse.ArgumentParser(description='mask rcnn')    
parser.add_argument('--file_name', required = True, help='folder name')
parser.add_argument('--model_dir', required = True, help='folder name')
parser.add_argument('--jpg_dir', required = True, help='folder name')
parser.add_argument('--seg_dir', required = True, help='folder name')
parser.add_argument('--token', type=str, required=True) 
args = parser.parse_args()

slack_token = args.token
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

for idx in range(len(files)):
    img = cv2.imread(os.path.join(jpg_dir,folder,'%d.jpg'%(idx)))
    img = img.astype(np.float32)
    img = img/255
    output = model(torch.tensor([img.transpose(2,0,1)]).to(device))
    try :
        cv2.imwrite(os.path.join(seg_dir,folder,'%d.png'%(idx)),
                    output[0]['masks'].detach().cpu().numpy()[0][0]*255)
    except:
        cv2.imwrite(os.path.join(seg_dir,folder,'%d.png'%(idx)),
                    np.zeros(img.shape[:2]))
    
token = slack_token
channel = "#finish-alarm"
text = folder+" inference finish"

requests.post("https://slack.com/api/chat.postMessage",
    headers={"Authorization": "Bearer "+token},
    data={"channel": channel,"text": text})