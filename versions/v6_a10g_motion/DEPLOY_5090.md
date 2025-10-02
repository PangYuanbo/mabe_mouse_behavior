# 远程5090主机部署指南

## 🎯 快速部署V6训练

### 1. 克隆仓库

```bash
# SSH到远程5090主机
ssh your-5090-host

# 克隆仓库
git clone <your-repo-url>
cd mabe_mouse_behavior

# 查看V6分支
git log --oneline -1
# 应该看到: Add V6 training version with Motion Features
```

### 2. 环境设置

```bash
# 创建conda环境（推荐）
conda create -n mabe python=3.11 -y
conda activate mabe

# 安装PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 验证GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 应该输出: CUDA: True, GPU: NVIDIA GeForce RTX 5090
```

### 3. 准备Kaggle数据

**选项A: 直接下载（推荐）**

```bash
# 设置Kaggle凭据
mkdir -p ~/.kaggle
nano ~/.kaggle/kaggle.json
# 粘贴你的API key:
# {"username":"your_username","key":"your_api_key"}

chmod 600 ~/.kaggle/kaggle.json

# 下载数据
pip install kaggle
kaggle competitions download -c MABe-mouse-behavior-detection

# 解压
mkdir -p data/kaggle
unzip MABe-mouse-behavior-detection.zip -d data/kaggle/
```

**选项B: 从本地rsync**

```bash
# 在本地执行
rsync -avz --progress data/kaggle/ your-5090-host:~/mabe_mouse_behavior/data/kaggle/
```

### 4. 开始训练

```bash
# 使用默认配置
python train_v6_local_5090.py

# 使用tmux/screen保持后台运行（推荐）
tmux new -s mabe_train
python train_v6_local_5090.py
# Ctrl+B, D 分离session

# 重新连接
tmux attach -t mabe_train
```

---

## 📊 监控训练

### 实时监控

```bash
# 查看checkpoint
watch -n 10 'ls -lh checkpoints/v6_5090/'

# 查看GPU使用
watch -n 2 nvidia-smi
```

### 预期性能

```bash
# 训练开始后，应该看到：
# Epoch 1/100: 100%|████████| Loss: 0.xxx, F1: 0.5xx
#
# 性能指标：
# - 每epoch: ~2-3分钟
# - GPU利用率: 80-95%
# - 显存使用: ~15-20 GB / 32 GB
# - 预计总时长: 3-5小时
```

---

## 🔧 配置调整

### config_5090.yaml 关键参数

```yaml
# 根据你的5090调整
batch_size: 96          # 可尝试 128
num_workers: 4          # 根据CPU核心数调整
learning_rate: 0.0003   # 如果batch增大，相应调整
```

### 如果OOM（显存不足）

```bash
# 编辑配置
nano configs/config_5090.yaml

# 修改
batch_size: 64  # 从96减到64
```

---

## 📁 重要文件

```
mabe_mouse_behavior/
├── train_v6_local_5090.py     # 主训练脚本 ← START HERE
├── configs/config_5090.yaml   # 5090配置
├── README_V6_5090.md          # 详细文档
│
├── data/kaggle/               # Kaggle数据目录
│   ├── train.csv
│   ├── train_tracking/
│   └── train_annotation/
│
├── checkpoints/v6_5090/       # Checkpoint保存位置
│   ├── latest_checkpoint.pth
│   ├── best_model.pth
│   └── checkpoint_epoch_*.pth
│
└── versions/v6_h100_current/  # V6文档和参考
    └── README.md
```

---

## 🎯 完整训练流程

```bash
# 1. 环境准备
conda activate mabe
cd ~/mabe_mouse_behavior

# 2. 验证GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# 3. 启动tmux
tmux new -s mabe_train

# 4. 开始训练
python train_v6_local_5090.py \
    --config configs/config_5090.yaml \
    --data-dir data/kaggle \
    --checkpoint-dir checkpoints/v6_5090

# 5. 分离tmux (Ctrl+B, D)
# 6. 断开SSH连接（训练继续）
exit

# 7. 稍后重新连接查看
ssh your-5090-host
tmux attach -t mabe_train
```

---

## 💾 训练完成后

### 下载最佳模型

```bash
# 在本地执行
scp your-5090-host:~/mabe_mouse_behavior/checkpoints/v6_5090/best_model.pth ./

# 查看性能
python evaluate_checkpoint.py --checkpoint best_model.pth
```

### 用于Kaggle提交

```bash
# 模型已经可以用于提交
# 参考: KAGGLE_SUBMISSION_GUIDE.md
```

---

## 🐛 常见问题

### 1. CUDA版本不匹配

```bash
# 检查CUDA版本
nvcc --version

# 重新安装对应PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # 或cu121
```

### 2. 数据加载慢

```yaml
# 增加workers
num_workers: 8  # 如果CPU核心充足
```

### 3. 训练中断恢复

```bash
# 自动从最新checkpoint恢复
python train_v6_local_5090.py --resume checkpoints/v6_5090/latest_checkpoint.pth
```

---

## 📈 性能对比

| 版本 | GPU | 输入维度 | F1 Score | 训练时间 |
|------|-----|----------|----------|----------|
| V5 | A10G | 144 | 0.43 | 12h |
| **V6** | **5090** | **288** | **~0.60+** | **3-5h** |

---

## 📞 支持

如有问题，查看：
- `README_V6_5090.md` - 完整使用指南
- `versions/v6_h100_current/README.md` - V6架构详解
- `VERSION_HISTORY.md` - 版本演进

---

**准备好了就开始训练！** 🚀

推荐命令：
```bash
tmux new -s mabe_train
python train_v6_local_5090.py
```
