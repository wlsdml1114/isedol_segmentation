import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse
import datetime
import requests
from collections import Counter
print(datetime.datetime.now())

def find_max_connected_component(img):

    temp = cv2.threshold(img,127,255,cv2.THRESH_BINARY)[1]

    labels, ims = cv2.connectedComponents(temp[:,:,0])

    res = Counter(ims.flatten())
    res.pop(0,None)
    max_key = max(res,key = res.get)

    ims[ims!=max_key] = 0
    ims[ims==max_key] = 255
    return ims

def get_args():
	
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	# backbone: the backbone of GFM, we provide four backbones - r34, r34_2b, d121 and r101.
	# rosta (Representations of Semantic and Transition areas): we provide three types - TT, FT, and BT. 
	# We also present RIM indicates RoSTa Integration Module.
	# model_path: path of the pretrained model to use
	# pred_choice(TT/FT/BT): 1 (glance decoder), 2 (focus decoder) and 3 (final result after Collaborative Matting) 
	# pred_choice(RIM): 1 (TT result), 2 (FT result), 3 (BT result), and 4 (RIM result). 
	# test_choice: test strategy (HYBRID or RESIZE)
	# test_result_dir: path to save the test results
	# logname: name of the logging files
	parser.add_argument('--file_name', type=str, required=True)
	parser.add_argument('--jpg_dir', type=str,required=False,default='/home/jini1114/git/MODNet/output')
	parser.add_argument('--png_dir', type=str, required=False,default='/home/jini1114/git/MODNet/temp')
	parser.add_argument('--seg_dir', type=str, required=False,default='/home/jini1114/git/MODNet/temp')
	parser.add_argument('--out_dir', type=str, required=False,default='/home/jini1114/git/MODNet/mp4')
	parser.add_argument('--fps', type=int, required=False,default=60)
	parser.add_argument('--token', type=str, required=True)
	args = parser.parse_args()
	return args


def scale_contour(cnt, scale):
    try :
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            cnt_scaled = cnt
        else :
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            cnt_norm = cnt - [cx, cy]
            cnt_scaled = cnt_norm * scale
            cnt_scaled = cnt_scaled + [cx, cy]
            cnt_scaled = cnt_scaled.astype(np.int32)
    except :
        arr = []
        for i in range(len(cnt)):

            M = cv2.moments(cnt[i])
            if M['m00'] == 0:
                cnt_scaled = cnt[i]
            else :
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])

                cnt_norm = cnt[i] - [cx, cy]
                cnt_scaled = cnt_norm * scale
                cnt_scaled = cnt_scaled + [cx, cy]
                cnt_scaled = cnt_scaled.astype(np.int32)
                arr.append(cnt_scaled)
        cnt_scaled = arr
    return cnt_scaled


args = get_args()

slack_token = args.token
file_name = args.file_name
file_names = file_name.split('/')
origin_file_name = file_names[-1]


jpg_dir = os.path.join(args.jpg_dir,origin_file_name)
png_dir = os.path.join(args.png_dir,origin_file_name)
seg_dir = os.path.join(args.seg_dir,origin_file_name)
out_dir = args.out_dir
fps = args.fps
'''
#mod
frames = []

files = os.listdir(png_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    modnet = cv2.imread(os.path.join(png_dir,'%d.png'%(idx)))
    gray = cv2.cvtColor(modnet, cv2.COLOR_RGB2GRAY)
    res = cv2.findContours(gray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[-2]
    contours = scale_contour(contours,0.9)
    cv2.drawContours(modnet, contours, contourIdx=-1, color=(255,255,255),thickness=-1)

    frames.append(modnet.astype(np.uint8))

h,w,l = modnet.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'mod_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()
'''
'''
#jpg
frames = []

files = os.listdir(png_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    modnet = cv2.imread(os.path.join(png_dir,'%d.png'%(idx)))
    frames.append(modnet)

h,w,l = modnet.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'mod_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()


#jpg
frames = []

files = os.listdir(png_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    temp = cv2.imread(os.path.join(png_dir,'%d.png'%(idx)))
    gray = temp[:,:,0]
    res = cv2.findContours(gray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[-2]
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    try:
        idx = np.where(np.max(area) == area)[0][0]
        cv2.drawContours(temp, contours, contourIdx=idx, color=(255,255,255),thickness=-1)
    except:
        pass
    frames.append(temp.astype(np.uint8))

h,w,c = temp.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'modful_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()
'''
#seg
frames = []

files = os.listdir(seg_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    modnet = cv2.imread(os.path.join(seg_dir,'%d.png'%(idx)))
    '''
    gray = cv2.cvtColor(modnet, cv2.COLOR_RGB2GRAY)
    res = cv2.findContours(gray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[-2]
    contours = scale_contour(contours,0.9)
    cv2.drawContours(modnet, contours, contourIdx=-1, color=(255,255,255),thickness=-1)
    '''
    frames.append(modnet.astype(np.uint8))

h,w,l = modnet.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'seg_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()


<<<<<<< Updated upstream
#seg
frames = []
THRESHOLD = 127

files = os.listdir(seg_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    modnet = cv2.imread(os.path.join(seg_dir,'%d.png'%(idx)))
    modnet[modnet>THRESHOLD] = 255
    modnet[modnet<=THRESHOLD] = 0
    frames.append(modnet.astype(np.uint8))

h,w,l = modnet.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'segthre_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)
=======
##

frames = []

kernel = np.ones((10,10),np.float32)/100

files = os.listdir(seg_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    temp = cv2.imread(os.path.join(seg_dir,'%d.png'%(idx)))
    temp[temp>125] = 255
    temp[temp<=125] = 0

    dst = cv2.filter2D(temp,-1,kernel)

    frames.append(dst)

h,w,l = dst.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'seg_filter_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)
>>>>>>> Stashed changes

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()

<<<<<<< Updated upstream
=======

##

frames = []

kernel = np.ones((10,10),np.float32)/100

files = os.listdir(seg_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    temp = cv2.imread(os.path.join(seg_dir,'%d.png'%(idx)))
    temp[temp>127] = 255
    temp[temp<=127] = 0
    try :
        maximg = find_max_connected_component(temp)
    except :
        maximg = temp
    dst = cv2.filter2D(temp,-1,kernel)

    frames.append(dst)

h,w,l = dst.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'seg_filter_component_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()
>>>>>>> Stashed changes

#seg_contour_full
frames = []

files = os.listdir(seg_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    temp = cv2.imread(os.path.join(seg_dir,'%d.png'%(idx)))
    gray = temp[:,:,0]
    res = cv2.findContours(gray.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = res[-2]
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    try:
        idx = np.where(np.max(area)==area)[0][0]
        cv2.drawContours(temp, contours, contourIdx=idx, color=(255,255,255),thickness=-1)
    except :
        pass
    frames.append(temp.astype(np.uint8))

h,w,c = temp.shape
size = (w,h)

output = cv2.VideoWriter(os.path.join(out_dir,'segfull_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output.write(frames[i])

output.release()





#cut origin
frames = []

files = os.listdir(jpg_dir)
for idx in tqdm(range(len(files)),desc = 'frame loading'):
    img = cv2.imread(os.path.join(jpg_dir,'%d.jpg'%(idx)))
    frames.append(img)
h,w,l = img.shape
size = (w,h)

output_cut = cv2.VideoWriter(os.path.join(out_dir,'cut_'+origin_file_name),cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

for i in tqdm(range(len(frames)),desc = 'mp4 making'):
    output_cut.write(frames[i])

output_cut.release()
token = slack_token
channel = "#finish-alarm"
text = origin_file_name+" png2mp4 finish"

requests.post("https://slack.com/api/chat.postMessage",
    headers={"Authorization": "Bearer "+token},
    data={"channel": channel,"text": text})

#os.system('rm -r '+jpg_dir)
#os.system('rm -r '+png_dir)
#os.system('rm -r '+seg_dir)