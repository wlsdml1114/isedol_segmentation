import os
import argparse

def get_args():
	
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--ori_dir', type=str, required=True)
	parser.add_argument('--wav_dir', type=str, required=True)
	parser.add_argument('--out_dir', type=str, required=True)
	return parser.parse_args()

args = get_args()

ori_dir = args.ori_dir
out_dir = args.out_dir
wav_dir = args.wav_dir

files = os.listdir(ori_dir)

for file in files :
    os.system('mv '+os.path.join(wav_dir,'results/baseline',file,'vocals.wav')+' '+os.path.join(out_dir,'%s_vocal.wav'%(file[:-4])))

os.system('rm -r '+os.path.join(wav_dir,'results/baseline/*'))
os.system('rm -r '+os.path.join(wav_dir,'test/*'))