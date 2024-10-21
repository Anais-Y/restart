#!/bin/bash

# 设置窗口长度
window_len=96  # 如果窗口长度是固定的，否则需要调整

# 获取格式化日期
formatted_date=$(date +"%m%d%H")
current_date=$(date +"%m%d")

# 设置要处理的数据文件名
declare -a subjects=("s16" "s15" "s19" "s09" "s30" "s32" "s27" "s26" "s21" "s18" "s11" "s31")

# 初始化计数器
count=1
parallel_limit=6  # 同时运行的最大任务数
active_jobs=0  # 当前活跃的后台任务数

# 遍历数据文件
for subject in "${subjects[@]}"; do
    data_path="/data/Anaiis/Data/DEAP/len_${window_len}/${subject}/"  # 假设数据路径格式正确
    description=$(basename "$data_path")

    # 在后台执行Python脚本
    (
        python wind_len_abla.py --config_file configs/DEAP/s01.conf \
        --hidden_dim 128 \
        --data "$data_path" --desc "$description" \
        --strides 3 \
        --window_length $window_len \
        --expid $count-$current_date > "/data/Anaiis/garage/$formatted_date-$description.txt"
        
        # 检查上一个命令的退出状态
        if [ $? -eq 0 ]; then
            echo "Successfully processed $description"
        else
            echo "Error processing $description"
        fi
    ) &

    # 更新后台任务计数
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

# 等待所有后台任务完成
wait
echo "All tasks completed."
