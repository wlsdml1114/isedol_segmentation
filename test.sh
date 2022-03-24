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
file_path='/home/jini1114/git/MODNet/input'
temp_dir='/home/jini1114/git/MODNet/output'
path_in='/home/jini1114/git/MODNet/temp'
gfm_out='/home/jini1114/git/GFM/samples/result_alpha'
path_out='/home/jini1114/git/MODNet/mp4'
fps=60

for file in "$file_path"/*
do
    echo "$file" task start
    
    /usr/anaconda3/envs/hair_task/bin/python /home/jini1114/git/MODNet/png2mp4.py\
        --file_path=$file_path \
        --file_name=$file \
        --temp_dir=$temp_dir \
        --path_in=$path_in \
        --gfm_out=$gfm_out \
        --path_out=$path_out \
        --fps=$fps 
done