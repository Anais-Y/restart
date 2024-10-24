#!/bin/bash

hidden_dims=(64 128 256)
num_heads=(2 4)
dropout_disactives=(0.3 0.6 0.9)
learning_rates=(1e-3 1e-4 5e-4)
log_path="/data/Anaiis/logs/SEED_"

formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")
count=1

for data_path in /data/Anaiis/Data/SEED_IV/len_200/1_20160518/; do
    for hidden_dim in "${hidden_dims[@]}"; do
         for num_head in "${num_heads[@]}"; do
            for dropout in "${dropout_disactives[@]}"; do
                for lr in "${learning_rates[@]}"; do
                    description=$(basename "$data_path")    
                    expid="hd${hidden_dim}_nh${num_head}_do${dropout}_lr${lr}"
                    python watch_grid_search_2.py \
                        --config_file configs/SEED/iv.conf \
                        --data $data_path \
                        --hidden_dim $hidden_dim \
                        --num_heads $num_head \
                        --desc "$description" \
                        --dropout_disactive $dropout \
                        --learning_rate $lr \
                        --expid $expid > /data/Anaiis/garage/onlyDE_seed_noshuf/$formatted_date-$description.txt \
                        --log_file ${log_path}${expid}.log \
                        --seed 1
                    if [ $? -eq 0 ]; then
                        echo "Successfully processed $description"
                    else
                        echo "Error processing $description"
                    fi
                    count=$((count + 1))
                done
            done
        done
    done
    
done
