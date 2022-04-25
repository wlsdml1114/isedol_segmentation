#/bin/bash

ori_dir='/data/codes/data/input'
out_dir='/data/codes/data/mp4'
wav_dir='/data/codes/mdx-net-submission/data'

/home/smartai/anaconda3/envs/mdx/bin/python /data/codes/isedol_segmentation/wav_demixing.py\
    --ori_dir=$ori_dir \
    --wav_dir=$wav_dir

/home/smartai/anaconda3/envs/mdx/bin/python /data/codes/mdx-net-submission/predict_blend.py

/home/smartai/anaconda3/envs/mdx/bin/python /data/codes/isedol_segmentation/move_wav.py\
    --ori_dir=$ori_dir \
    --wav_dir=$wav_dir \
    --out_dir=$out_dir