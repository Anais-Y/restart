#!/bin/bash
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/DEAP/len_96/s*/; do
    description=$(basename "$data_path")
    python watch_abla_3.py --config_file configs/DEAP/s01.conf \
    --data $data_path --desc "$description" \
    --strides 3 \
    --hidden_dim 64\
    --save /data/Anaiis/garage/Mix_deap_shuf_100/\
    --expid $count-$current_date > /data/Anaiis/garage/Mix_deap_shuf_100/$formatted_date-$description.txt &

    if [ $((count % 2)) -eq 0 ]; then
        wait # 等待所有后台任务完成
    fi
    
    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi

    count=$((count + 1))  # 更新计数器
done

wait
