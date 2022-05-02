#/bin/bash

ori_dir='/home/jini1114/git/data/input'
out_dir='/home/jini1114/git/data/mp4'
wav_dir='/home/jini1114/git/mdx-net-submission/data'

/home/jini1114/.conda/envs/mdx-net/bin/python /home/jini1114/git/isedol_segmentation/wav_demixing.py\
    --ori_dir=$ori_dir \
    --wav_dir=$wav_dir

/home/jini1114/.conda/envs/mdx-net/bin/python /home/jini1114/git/mdx-net-submission/predict_blend.py

/home/jini1114/.conda/envs/mdx-net/bin/python /home/jini1114/git/isedol_segmentation/move_wav.py\
    --ori_dir=$ori_dir \
    --wav_dir=$wav_dir \
    --out_dir=$out_dir