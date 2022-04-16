import os
import subprocess
import argparse

def get_args():
	
	parser = argparse.ArgumentParser(description='Arguments for the testing purpose.')	
	parser.add_argument('--ori_dir', type=str, required=True)
	parser.add_argument('--wav_dir', type=str, required=True)
	args = parser.parse_args()
	return args

args = get_args()

ori_dir = args.ori_dir
wav_dir = args.wav_dir

print('mp4 to wav')

files = os.listdir(ori_dir)

for file in files:
	if not(os.path.exists(os.path.join(wav_dir,'test',file))) : 
		os.system('mkdir -p ' + os.path.join(wav_dir,'test',file))
	command = "ffmpeg -y -i {} -ac 2 -f wav {}".format(os.path.join(ori_dir,file), os.path.join(wav_dir,'test',file, 'mixture.wav'))
	subprocess.call(command, shell=True)