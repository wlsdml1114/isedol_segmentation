#/bin/bash

###########################
## Arch and model_path 
###########################
# backbone: r34 | rosta: TT | model_path: models/pretrained/gfm_r34_tt.pth
# backbone: r34 | rosta: FT | model_path: models/pretrained/gfm_r34_ft.pth
# backbone: r34 | rosta: BT | model_path: models/pretrained/gfm_r34_bt.pth
# backbone: r34_2b | rosta: TT | model_path: models/pretrained/gfm_r34_2b_tt.pth
# backbone: d121 | rosta: TT | model_path: models/pretrained/gfm_d121_tt.pth
# backbone: r101 | rosta: TT | model_path: models/pretrained/gfm_r101_tt.pth

backbone='r34'
rosta='TT'
model_path='/home/jini1114/git/GFM/models/pretrained/gfm_r34_tt.pth'
dataset_choice='SAMPLES'
test_choice='HYBRID'
pred_choice=3
ori_dir='/home/jini1114/git/data/input'
jpg_dir='/home/jini1114/git/data/output'
png_dir='/home/jini1114/git/data/temp'
tr_data_dir='/home/jini1114/git/data/dataset/data'
seg_dir='/home/jini1114/git/data/segmentation'
model_dir='/home/jini1114/git/data/model'
aug_dir='/home/jini1114/git/data/augmentation'
out_dir='/home/jini1114/git/data/mp4'
fps=60

for file in "$ori_dir"/*
do
    echo "$file" task start
    
    /usr/anaconda3/envs/hair_task/bin/python /home/jini1114/git/isedol_segmentation/faster-r-cnn.py\
        --name=$file \
        --aug_dir=$aug_dir \
        --model_dir=$model_dir
done