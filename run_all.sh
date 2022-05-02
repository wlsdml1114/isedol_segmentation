#/bin/bash

ori_dir='/home/jini1114/git/data/input'

nohup bash demixing.sh &

for file in "$ori_dir"/*
do
    nohup bash matte.sh $file &
done