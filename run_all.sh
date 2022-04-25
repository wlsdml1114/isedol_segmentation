#/bin/bash

ori_dir='/data/codes/data/input'

nohup bash demixing.sh &

for file in "$ori_dir"/*
do
    nohup bash matte.sh $file &
done