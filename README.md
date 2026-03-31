# SL-YOLO: An Efficient Real-Time Hand Gesture Detection Network Using Structured Convolution and Re-parameterized Large Kernels

## Model Architecture
SL-YOLO is built on YOLOv8 with the following improvements:
- Lightweight backbone optimization
-Efficient convolution via DSConv
-Large kernel modeling via C2RepLK
-Stable feature refinement via LSKblock


## Datasets

The experiments are conducted on multiple datasets:

1. ASL Dataset (Sign Language)

🔗 https://universe.roboflow.com/meredith-lo-pmqx7/asl-project

2. Expression Dataset (Hand Gestures)

🔗 https://universe.roboflow.com/expression/expressions-tgbkg

3. Kidney Tumor Dataset (Generalization)

🔗 https://universe.roboflow.com/tezskb/kidney-tumor-uqpis



### 3. Install Dependencies
(It is recommended to directly use the YOLOv11 or YOLOv8 environment that has already been set up on this computer, without the need to download again.)
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


### 4. Run the Program
```bash
python train.py --data your_dataset_config.yaml
```
#### Explanation of Training Modes

Below are the Python script files for different training modes included in the project, each targeting specific training needs and data types.

4.1. **`train.py`**
   - Basic training script.
   - Used for standard training processes, suitable for general image classification or detection tasks.

2. **`train-rtdetr.py`**
   - Training script for RTDETR (Real-Time Detection Transformer).

3. **`train_Gray.py`**
   - Grayscale image training script.
   - Specifically for processing datasets of grayscale images, suitable for tasks requiring image analysis in grayscale space.


### 5. Testing
Run the test script to verify if the data loading is correct:
```bash
python val.py
```
