# AI_Planner 环境安装总结 (x86 Laptop + RTX 3060)

## 1. 环境初始化

```
conda create -n AI_Planner python=3.10 -y
conda activate AI_Planner
```

## 2. 核心依赖安装

```
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

## 3. GraspNet 模块安装

### 3.1 下载Graspnet源码

```
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline-main
```

### 3.2 编辑 requirements.txt为：

```
numpy==1.23.0
scipy
open3d>=0.8
Pillow
tqdm
```

### 3.3 安装相关依赖

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 编译安装pointnet2和knn

### 4.1 安装pointnet2

```
cd pointnet2
python setup.py install
```

### 4.2 安装KNN

```
cd knn
python setup.py install
```

## 5. 安装graspnetAPI 

### 5.1 下载graspnetAPI 代码

```
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
```

### 5.2 修改setup.py文件

```
需要修改 setup.py文件，将其中的sklearn替换为scikit-learn，并且numpy==1.23.0
```

### 5.3 安装

```
pip install . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

