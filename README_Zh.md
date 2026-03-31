
# SL-YOLO：基于结构化卷积与重参数化大核的高效实时手语检测网络





## 模型架构
SL-YOLO 基于 YOLOv8 构建，并进行了如下改进：
-轻量化骨干网络优化
-基于 DSConv 的高效卷积设计
-基于 C2RepLK 的大感受野建模
基于 LSKblock 的稳定特征增强


## 数据集

本项目在多个数据集上进行实验验证：

1. ASL 手语数据集

🔗 https://universe.roboflow.com/meredith-lo-pmqx7/asl-project

2. Expression 手势数据集

🔗 https://universe.roboflow.com/expression/expressions-tgbkg

3. 肾脏肿瘤数据集（泛化验证）

🔗 https://universe.roboflow.com/tezskb/kidney-tumor-uqpis



### 3. Install Dependencies
(环境安装推荐直接使用已配置好的 YOLOv8 或 YOLOv11 环境，无需重复安装）
```bash
# Step 1.Create a virtual environment with conda
conda create -n pt121_py38 python=3.8
conda activate pt121_py38

# Step 2: Install pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch


# Step 3: Install the remaining dependencies

pip install -r requirements.txt


# https://pytorch.org/get-started/previous-versions/
## CUDA 10.2
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
## CUDA 11.3
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
## CUDA 11.6
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
## CPU Only
#conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch

## CUDA 11.8
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
## CUDA 12.1
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
## CPU Only
#conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 cpuonly -c pytorch
```


### 4. 运行训练
```bash
python train.py --data your_dataset_config.yaml
```
#### 训练脚本说明

本项目包含多个训练脚本，适用于不同任务：

4.1. **`train.py`**
  - 基础训练脚本，适用于通用目标检测任务


4.2. **`train-rtdetr.py`**
   - 用于 RT-DETR 模型的训练

4.3. **`train_Gray.py`**
   - 灰度图训练脚本，适用于单通道图像任务


### 5.测试与验证

运行以下命令进行模型验证：
```bash
python val.py
```
