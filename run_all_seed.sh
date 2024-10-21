#!/bin/bash
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/SEED_IV/len_200/*/; do
    description=$(basename "$data_path")    
    python watch_PE.py --config_file configs/SEED/iv.conf\
    --hidden_dim 256\
    --data $data_path --desc "$description" \
    --strides 3\
    --expid $count-$current_date > /data/Anaiis/garage/SEEDiv_shuf/$formatted_date-$description.txt
    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi
    count=$((count + 1))  # 更新计数器
done