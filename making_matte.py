import os
import cv2
from tqdm import tqdm
import argparse
from demo.image_matting.colab.inference2 import run
import datetime
import requests
import subprocess
print(datetime.datetime.now())


def get_args():
	
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--ori_dir', type=str, required=True)
	parser.add_argument('--file_name', type=str, required=True)
	parser.add_argument('--jpg_dir', type=str,required=True)
	parser.add_argument('--png_dir', type=str, required=True)
	parser.add_argument('--token', type=str, required=True)
	parser.add_argument('--wav_dir', type=str, required=True)
	args = parser.parse_args()
	return args


args = get_args()

slack_token = args.token
file_name = args.file_name
file_names = file_name.split('/')
origin_file_name = file_names[-1]

ori_dir = args.ori_dir
wav_dir = os.path.join(args.wav_dir,'test',origin_file_name)
jpg_dir = os.path.join(args.jpg_dir,origin_file_name)
png_dir = os.path.join(args.png_dir,origin_file_name)


print(origin_file_name,'start')
if not(os.path.exists(jpg_dir)) : 
    os.system('mkdir -p ' + jpg_dir)
if not(os.path.exists(png_dir)) : 
    os.system('mkdir -p ' + png_dir)

cap = cv2.VideoCapture(os.path.join(ori_dir,origin_file_name))

count = 0
while True :

    if (count%1000 == 0):
        print(count)
    ret, frame = cap.read()

    if ret == False :
        break

    if origin_file_name[0] == 'l' :
        cv2.imwrite(os.path.join(jpg_dir,'%d.jpg'%(count)),frame[400:,:800,:])
    if origin_file_name[0] == 'r' :
        cv2.imwrite(os.path.join(jpg_dir,'%d.jpg'%(count)),frame[400:,1200:,:])
    count+=1


run(jpg_dir,png_dir)

token = slack_token
channel = "#finish-alarm"
text = origin_file_name+" making matte finish"

requests.post("https://slack.com/api/chat.postMessage",
    headers={"Authorization": "Bearer "+token},
    data={"channel": channel,"text": text})