#!/bin/bash
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/DEAP/s*/; do
    description=$(basename "$data_path")    
    python watch_PE.py --config_file configs/DEAP/s01.conf --data $data_path --desc "$description" --expid $count-$current_date > /data/Anaiis/garage/deap_topK/$formatted_date-$description.txt
    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi
    count=$((count + 1))  # 更新计数器
done