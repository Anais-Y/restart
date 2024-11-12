#!/bin/bash

BASE_DIR="/data/Anaiis/anti_overfit/wandb"

for dir in $BASE_DIR/offline*; do
    if [ -d "$dir" ]; then  # 确保是一个目录
        echo "Syncing $dir..."
        wandb sync "$dir" 
    fi
done

echo "All syncs completed."
