#!/bin/bash
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/SEED/len_50/smooth_False/*/; do
    description=$(basename "$data_path")
    python watch_PE_cp.py --config_file configs/SEED/1.conf \
    --window_length 50 \
    --data $data_path --desc "$description" \
    --strides 1 \
    --hidden_dim 64\
    --save /data/Anaiis/garage/len50_seed_step1-2/\
    --expid $count-$current_date > /data/Anaiis/garage/len50_seed_step1-2/$formatted_date-$description.txt &

    if [ $((count % 15)) -eq 0 ]; then
        wait # 等待所有后台任务完成
    fi

    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi
    count=$((count + 1))  # 更新计数器
done