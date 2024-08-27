#!/bin/bash

count=1
mkdir outputs_august
for data_path in /data/Anaiis/Data/Data/SEED/len_200/smooth_False/*/; do
    description=$(basename "$data_path")    
    python watch_PE.py --config_file configs/SEED/1.conf --data $data_path --desc "$description" --expid $count > outputs_august/0827_$description.txt
    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi
    count=$((count + 1))  # 更新计数器
done