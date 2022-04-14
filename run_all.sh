#/bin/bash

ori_dir='/home/jini1114/git/data/input'

for file in "$ori_dir"/*
do
    nohup bash matte.sh $file &
done