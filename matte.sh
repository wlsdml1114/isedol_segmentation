#/bin/bash

ori_dir='/data/codes/data/input'
jpg_dir='/data/codes/data/output'
seg_dir='/data/codes/data/segmentation'
png_dir='/data/codes/data/temp'
tr_data_dir='/data/codes/data/dataset'
out_dir='/data/codes/data/mp4'
model_dir='/data/codes/data/model'
fps=60

file=$1
echo "$file" task start

/home/smartai/anaconda3/envs/seg/bin/python /data/codes/isedol_segmentation/making_matte.py\
    --ori_dir=$ori_dir \
    --file_name=$file \
    --png_dir=$png_dir \
    --jpg_dir=$jpg_dir \
    --token=$SLACK_TOKEN
    
/home/smartai/anaconda3/envs/seg/bin/python /data/codes/isedol_segmentation/inference.py\
    --seg_dir=$seg_dir \
    --file_name=$file \
    --jpg_dir=$jpg_dir \
    --model_dir=$model_dir \
    --token=$SLACK_TOKEN

/home/smartai/anaconda3/envs/seg/bin/python /data/codes/isedol_segmentation/png2mp4.py\
    --file_name=$file \
    --jpg_dir=$jpg_dir \
    --png_dir=$png_dir \
    --out_dir=$out_dir \
    --fps=$fps \
    --seg_dir=$seg_dir \
    --token=$SLACK_TOKEN
    