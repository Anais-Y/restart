# Restart for graduate

### 20240827进度
对于DEAP数据集:
    每个被试全打乱的话效果奇佳
    不打乱validation loss不下降，具体可以看wandb 0824-deap-noshuf的训练记录
对于SEED数据集
    每个被试打乱：如果完完全全按照DEAP的方法过拟合，训练集到1测试集到60就不上升了--> 缺少数据
    - 加入DE特征，双塔，单个被试效果可以
    - 将DE特征作为一个图放在数据里，同样也可以
    每个被试不打乱，将DE特征作为一个图放在数据里

what about 跨人？
