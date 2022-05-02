#/bin/bash

ori_dir='/home/jini1114/git/data/input'
jpg_dir='/home/jini1114/git/data/output'
seg_dir='/home/jini1114/git/data/segmentation'
png_dir='/home/jini1114/git/data/temp'
tr_data_dir='/home/jini1114/git/data/dataset'
out_dir='/home/jini1114/git/data/mp4'
model_dir='/home/jini1114/git/data/model'
fps=60

file=$1
echo "$file" task start

/usr/anaconda3/envs/hair_task/bin/python /home/jini1114/git/isedol_segmentation/making_matte.py\
    --ori_dir=$ori_dir \
    --file_name=$file \
    --png_dir=$png_dir \
    --jpg_dir=$jpg_dir \
    --token=$SLACK_TOKEN
    
/usr/anaconda3/envs/hair_task/bin/python /home/jini1114/git/isedol_segmentation/inference.py\
    --seg_dir=$seg_dir \
    --file_name=$file \
    --jpg_dir=$jpg_dir \
    --model_dir=$model_dir \
    --token=$SLACK_TOKEN

/usr/anaconda3/envs/hair_task/bin/python /home/jini1114/git/isedol_segmentation/png2mp4.py\
    --file_name=$file \
    --jpg_dir=$jpg_dir \
    --png_dir=$png_dir \
    --out_dir=$out_dir \
    --fps=$fps \
    --seg_dir=$seg_dir \
    --token=$SLACK_TOKEN
    