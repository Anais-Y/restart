#!/bin/bash
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/Data/SEED/len_200/smooth_False/*/; do
    description=$(basename "$data_path")    
    python main.py --data $data_path --desc "$description" --expid $count-$current_date > /data/Anaiis/garage/onlyDE_seed_noshuf/$formatted_date-$description.txt
    if [ $? -eq 0 ]; then
        echo "Successfully processed $description"
    else
        echo "Error processing $description"
    fi
    count=$((count + 1)) 
done