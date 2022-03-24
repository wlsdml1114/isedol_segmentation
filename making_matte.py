import os
import cv2
from tqdm import tqdm
import argparse
from demo.image_matting.colab.inference2 import run


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
	parser.add_argument('--ori_dir', type=str, required=False,default='/home/jini1114/git/MODNet/input')
	parser.add_argument('--file_name', type=str, required=True)
	parser.add_argument('--jpg_dir', type=str,required=False,default='/home/jini1114/git/MODNet/output')
	parser.add_argument('--seg_dir', type=str,required=False,default='/home/jini1114/git/MODNet/output')
	parser.add_argument('--png_dir', type=str, required=False,default='/home/jini1114/git/MODNet/temp')
	args = parser.parse_args()
	return args


args = get_args()


file_name = args.file_name
file_names = file_name.split('/')
origin_file_name = file_names[-1]

ori_dir = args.ori_dir
jpg_dir = os.path.join(args.jpg_dir,origin_file_name)
png_dir = os.path.join(args.png_dir,origin_file_name)
seg_dir = os.path.join(args.seg_dir,origin_file_name)


if not(os.path.exists(jpg_dir)) : 
    os.system('mkdir -p ' + jpg_dir)
print(origin_file_name,'start')
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

