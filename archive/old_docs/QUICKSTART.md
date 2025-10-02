# 快速启动指南

## 30秒快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 生成示例数据（用于测试）
python create_sample_data.py

# 3. 本地训练（高级版）
python train_advanced.py --config configs/config_advanced.yaml

# 4. Modal 云端训练
modal setup  # 首次使用需要认证
modal run upload_data_to_modal.py
modal run upload_code_to_modal.py
modal run modal_train_advanced.py
```

## 使用真实 Kaggle 数据

### 1. 配置 Kaggle API

```bash
# 访问 https://www.kaggle.com/settings/account
# 点击 "Create New Token" 下载 kaggle.json

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. 下载竞赛数据

```bash
pip install kaggle
kaggle competitions download -c mabe-mouse-behavior-detection -p data
cd data
unzip mabe-mouse-behavior-detection.zip
```

### 3. 更新配置

编辑 `configs/config_advanced.yaml`:
```yaml
train_data_dir: 'data/train'
val_data_dir: 'data/val'
# 根据实际数据更新 input_dim 和 num_classes
```

### 4. 训练

```bash
# 本地训练
python train_advanced.py --config configs/config_advanced.yaml

# 或者 Modal 训练（推荐）
modal run upload_data_to_modal.py  # 上传真实数据
modal run modal_train_advanced.py
```

## Modal 云端训练（推荐）

### 优势
✅ 免费 A10G GPU（每月$30 credit）
✅ 无需本地 GPU
✅ 自动保存检查点
✅ 6小时超时（可调整）

### 步骤

```bash
# 1. 安装和认证
pip install modal
modal setup  # 浏览器中认证

# 2. 切换 workspace（如需要）
modal profile activate ybpang-1

# 3. 上传数据和代码
modal run upload_data_to_modal.py
modal run upload_code_to_modal.py

# 4. 开始训练
modal run modal_train_advanced.py

# 5. 下载模型
modal run modal_train_advanced.py::download_checkpoint > best_model.pth
```

### 监控训练

```bash
# 实时查看日志
modal app logs mabe-advanced --follow

# 查看历史日志
modal app logs mabe-advanced
```

## Jupyter Notebook

### 本地运行

```bash
jupyter lab notebooks/mabe_starter.ipynb
```

### Modal 运行

```bash
# 启动 Jupyter 服务器
modal run modal_notebook.py::run_jupyter

# 访问显示的 URL（通常是 https://your-app.modal.run）
```

## 模型选择

在 `config_advanced.yaml` 中设置：

```yaml
# 1. Conv1DBiLSTM（推荐，96% 准确率）
model_type: 'conv_bilstm'
conv_channels: [64, 128, 256]
lstm_hidden: 256
lstm_layers: 2

# 2. Temporal Convolutional Network
model_type: 'tcn'
tcn_channels: [64, 128, 256, 256]

# 3. Hybrid (PointNet + LSTM)
model_type: 'hybrid'
pointnet_dim: 128
temporal_model: 'lstm'  # 或 'transformer'
```

## 关键配置

### 特征工程

```yaml
use_feature_engineering: true  # 131维特征
include_pca: true              # PCA降维
include_temporal_stats: true   # 时间统计
pca_components: 16             # PCA成分数
```

### 数据增强

```yaml
use_augmentation: true
noise_std: 0.01          # 高斯噪声
temporal_jitter: 2       # 时间抖动
mixup_alpha: 0.2         # Mixup强度
```

### 训练策略

```yaml
# 处理类别不平衡
class_weights: [0.5, 2.0, 3.0, 3.0]

# 正则化
label_smoothing: 0.1
dropout: 0.3
weight_decay: 0.0001

# 学习率
learning_rate: 0.0005
scheduler: 'plateau'
scheduler_patience: 5

# 早停
early_stopping_patience: 15
```

## 性能调优

### 提升准确率

1. **增加序列长度**: `sequence_length: 100` → `150`
2. **增强特征工程**: `include_temporal_stats: true`
3. **使用类别权重**: 根据数据分布调整
4. **增加模型容量**: `hidden_dim: 256` → `512`

### 加速训练

1. **增大 batch size**: `batch_size: 32` → `64`
2. **减少序列长度**: `sequence_length: 100` → `64`
3. **使用更大 GPU**: Modal 改用 A100
4. **减少特征**: `include_temporal_stats: false`

### 防止过拟合

1. **增加 dropout**: `dropout: 0.3` → `0.5`
2. **数据增强**: `mixup_alpha: 0.2`, `noise_std: 0.02`
3. **早停**: `early_stopping_patience: 10`
4. **正则化**: `weight_decay: 0.0001` → `0.001`

## 常见问题

### Q: 内存不足

**A**: 减小 batch size 或序列长度
```yaml
batch_size: 16  # 从32减少
sequence_length: 64  # 从100减少
```

### Q: 训练太慢

**A**: 使用 Modal A100 GPU
```python
# modal_train_advanced.py
@app.function(
    gpu="A100",  # 改为 A100
    ...
)
```

### Q: 准确率不提升

**A**: 检查类别分布，调整类别权重
```python
# 分析数据
from collections import Counter
labels = ...
Counter(labels)  # 查看分布

# 调整权重（反比例）
class_weights = [1.0/freq for freq in frequencies]
```

### Q: 如何提交 Kaggle

**A**: 生成预测并提交
```python
# 在 notebook 中
# 1. 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 2. 对测试集预测
predictions = model(test_data)

# 3. 保存为 submission.csv
# 格式参考 Kaggle 要求
```

## 项目结构速查

```
├── configs/               # 配置文件
│   └── config_advanced.yaml    # ⭐ 主配置
├── src/                  # 源代码
│   ├── data/            # 数据处理
│   ├── models/          # 模型架构
│   └── utils/           # 训练工具
├── notebooks/            # Jupyter notebooks
├── data/                # 数据目录
├── checkpoints/         # 模型保存
├── train_advanced.py    # ⭐ 本地训练
├── modal_train_advanced.py  # ⭐ Modal训练
└── *.md                # 文档
```

## 下一步

1. **获取真实数据**: 从 Kaggle 下载
2. **探索数据**: 使用 Jupyter notebook
3. **调整配置**: 根据数据特点优化
4. **训练模型**: Modal 或本地训练
5. **生成提交**: 预测测试集
6. **迭代改进**: 分析结果，优化模型

## 获取帮助

- **文档**: `README.md`, `OPTIMIZATION_SUMMARY.md`
- **示例**: `notebooks/mabe_starter.ipynb`
- **配置**: `configs/config_advanced.yaml`
- **Kaggle**: https://www.kaggle.com/c/mabe-mouse-behavior-detection
- **Modal 文档**: https://modal.com/docs

## 关键命令速查

```bash
# 数据
python create_sample_data.py

# 本地训练
python train_advanced.py

# Modal 训练
modal run modal_train_advanced.py

# 上传数据/代码
modal run upload_data_to_modal.py
modal run upload_code_to_modal.py

# Jupyter
jupyter lab notebooks/mabe_starter.ipynb
modal run modal_notebook.py::run_jupyter

# 下载模型
modal run modal_train_advanced.py::download_checkpoint
```

---

**准备好了吗？开始训练吧！** 🚀

```bash
modal run modal_train_advanced.py
```