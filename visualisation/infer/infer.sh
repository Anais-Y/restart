#!/bin/bash

# CSV文件路径
csv_file="seed_50_shuf.csv"

# 跳过第一行（表头），然后读取每一行
tail -n +2 "$csv_file" | while IFS=, read -r line; do
    # 使用awk获取data列、exp_id列和val_loss列
    data=$(echo "$line" | awk -F, '{print $12}' | tr -d '"')
    exp_id=$(echo "$line" | awk -F, '{print $17}' | tr -d '"')
    val_loss=$(echo "$line" | awk -F, '{print $38}' | tr -d '"')
    
    # 四舍五入val_loss至两位小数
    rounded_val_loss=$(printf "%.2f" "$val_loss")
    
    # 检查 exp_id 是否包含多个 ID，如果是则拆分
    IFS=',' read -ra exp_ids <<< "$exp_id"

    # 循环遍历每个 exp_id
    for id in "${exp_ids[@]}"; do
        # 构造 checkpoint 文件名
        ckpt="/data/Anaiis/garage/len50_seed2/exp_${id}_0.0_best_model.pth"
        
        # 输出信息或执行操作
        echo "Processing checkpoint: $ckpt"
        
        # 检查文件是否存在
        if [ -f "$ckpt" ]; then
            echo "Checkpoint found: $ckpt"
            echo $data
            echo $ckpt
            echo $rounded_val_loss
            # 运行Python脚本，确保引号只用于变量展开
            python test_infer.py --data "$data" --ckpt "$ckpt"  
            
            # 检查命令执行状态
            if [ $? -eq 0 ]; then
                echo "Successfully processed $data"
            else
                echo "Error processing $data"
            fi
        else
            echo "Checkpoint not found: $ckpt"
        fi
    done
done
