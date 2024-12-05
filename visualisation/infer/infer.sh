#!/bin/bash

# CSV文件路径
csv_file="1126.csv"

# 跳过第一行（表头），然后读取每一行
tail -n +2 "$csv_file" | while IFS=, read -r line; do
    # 使用awk获取data列、exp_id列和val_loss列
    data=$(echo "$line" | awk -F, '{print $12}' | tr -d '"')
    exp_id=$(echo "$line" | awk -F, '{print $17}' | tr -d '"')
    # val_loss=$(echo "$line" | awk -F, '{print $38}' | tr -d '"')
    
    # 四舍五入val_loss至两位小数
    # rounded_val_loss=$(printf "%.2f" "$val_loss")
    
    # 检查 exp_id 是否包含多个 ID，如果是则拆分
    IFS=',' read -ra exp_ids <<< "$exp_id"

    # 循环遍历每个 exp_id
    for id in "${exp_ids[@]}"; do
        # 构造 checkpoint 文件的路径模式
        ckpt_pattern="/data/Anaiis/garage/len50_seed/exp_${id}_*_best_model.pth"
        
        # 遍历符合模式的所有文件
        for ckpt in $ckpt_pattern; do
            # 检查文件是否存在
            if [ -f "$ckpt" ]; then
                echo "Processing checkpoint: $ckpt"
                
                # 运行Python脚本
                python test_infer.py --data "$data" --ckpt "$ckpt"
                
                # 检查命令执行状态
                if [ $? -eq 0 ]; then
                    echo "Successfully processed $data with $ckpt"
                else
                    echo "Error processing $data with $ckpt"
                fi
            else
                echo "No checkpoint found matching pattern: $ckpt_pattern"
                break
            fi
        done
    done
done
