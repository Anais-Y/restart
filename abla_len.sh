#!/bin/bash

# 定义窗口长度数组
window_lengths=96

# 获取格式化日期
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")

# 对于每个窗口长度并行执行命令
for window_len in "${window_lengths[@]}"; do
    count=1
    parallel_limit=6  # 同时运行的最大任务数
    active_jobs=0
    # 遍历指定窗口长度的所有数据文件
    for data_path in /data/Anaiis/Data/DEAP/len_${window_len}/s*/; do
        description=$(basename "$data_path")
        
        # 执行Python脚本
        (
            python wind_len_abla.py --config_file configs/DEAP/s01.conf \
            --hidden_dim 128 \
            --data "$data_path" --desc "$description" \
            --strides 3 \
            --window_length $window_len \
            --expid $count-$current_date > /data/Anaiis/garage/$formatted_date-$description.txt
            
            # 检查上一个命令的退出状态
            if [ $? -eq 0 ]; then
                echo "Successfully processed $description"
            else
                echo "Error processing $description"
            fi
        ) &

        ((active_jobs++))
    
        # 检查是否达到并行限制
        if [[ $active_jobs -ge $parallel_limit ]]; then
            # 等待任一后台任务完成
            wait -n
            # 减少活跃任务计数
            ((active_jobs--))
        fi

        # 更新总任务计数器
        ((count++))
    done
done

# 等待所有后台进程完成
wait
echo "All processing done."
