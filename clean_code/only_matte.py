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
	parser.add_argument('--ori_dir', type=str, required=False,default = '/home/jini1114/git/data/input')
	parser.add_argument('--jpg_dir', type=str,required=False,default = '/home/jini1114/git/data/output')
	parser.add_argument('--png_dir', type=str, required=False,default = '/home/jini1114/git/data/temp')
	args = parser.parse_args()
	return args


args = get_args()

origin_file_name = 'r_cut_lilpa.mp4'

ori_dir = args.ori_dir
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