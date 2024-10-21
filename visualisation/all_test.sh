#!/bin/bash
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/DEAP/len_96/s*/; do
    python test.py \
    --data $data_path\
    --strides 3\
    --ckpt /data/Anaiis/garage/final-deap/exp_$count-*_best_model.pth
    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi
    count=$((count + 1))  # 更新计数器
done