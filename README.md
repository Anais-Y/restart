# **LISGA: 一种轻量级可解释的基于EEG的时空图注意力网络**

这是我毕业论文第四章的代码，同时也是LISGA: A Lightweight Interpretable Spatiotemporal Graph
Attention network for EEG-based Emotion Recognition这篇文章(未发表）的源码。

## Model Architecture

![Model Architecture](./images/structure.pdf)

## **项目结构**

### **主要文件夹**
- **`Effect_Att/`**  
  包含 EC-GraphNet 的代码，这是 LISGA 模型的前身，对应论文第三章的内容。详细信息请参考：[EC-GraphNet 仓库](https://github.com/Anais-Y/EC-GraphNet)。
  
- **`configs/`**  
  包含运行模型所需的配置文件，定义了超参数、数据集路径及其他运行时设置。

- **`data_process/`**  
  包含处理原始 EEG 数据的代码，将数据预处理为模型可用的格式。

- **`visualisation/`**  
  包含与模型可视化和推理相关的代码和结果。

---

### **可忽略的文件夹**
- **`new_try/`**  
  一次新的尝试，但未成功。
  
- **`outputs_august/`**  
  2024 年 8 月运行代码时生成的日志文件。

---

### **关键文件**
- **`watch**` 开头的文件**  
  用于运行模型并将结果记录在 `wandb` 平台上。如果不需要使用 `wandb`，可以使用 `main**` 开头的文件替代。

- **`utils_db.py`**  
  存放了项目中使用的一些工具函数。

- **`model*` 开头的文件**  
  - **`model_gat_seed.py`** 是论文中提出的 LISGA 模型实现。  
  - 其他 `model*` 文件是用于消融实验的模型变体。

- **`run_*.sh`**  
  用于批量运行实验的 Bash 脚本。

---

