# V3 - Modal云端基础版本

## 📋 版本概述

**版本号**: V3
**开发时间**: V2之后
**状态**: 已升级至V4
**目标**: 迁移至Modal云平台，实现GPU加速训练

---

## 🎯 设计思想

### 核心目标
1. 从本地迁移至Modal云平台
2. 利用云端GPU加速训练
3. 验证Modal部署流程

### V2 → V3 主要改进
- ✅ **云端部署**: 本地 → Modal云平台
- ✅ **GPU资源**: 本地GPU → Modal T4/A10G
- ✅ **可扩展性**: 固定资源 → 按需分配
- ✅ **持久化**: 本地存储 → Modal Volume

### 技术选择
- **平台**: Modal (workspace: ybpang-1)
- **GPU**: T4 (16GB VRAM)
- **数据**: 仍使用合成数据
- **模型**: Transformer（测试云端性能）
- **存储**: Modal Volume持久化

---

## 🏗️ Modal架构

### Modal函数定义

```python
import modal

app = modal.App("mabe-training-basic")

# 定义容器镜像
image = (modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "pyyaml",
        "tqdm",
        "scikit-learn"
    ))

# 定义持久化Volume
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

@app.function(
    image=image,
    gpu="T4",  # 16GB VRAM
    volumes={"/vol": volume},
    timeout=3600 * 2,  # 2小时超时
)
def train_model():
    """云端训练函数"""
    import torch
    from src.trainers.trainer import Trainer
    from src.models.transformer import TransformerModel

    # 训练逻辑
    trainer = Trainer(config)
    trainer.train()

    # 保存checkpoint到Volume
    volume.commit()
```

### 部署流程

```bash
# 1. 安装Modal
pip install modal

# 2. 配置Token
modal token new

# 3. 选择workspace
# 选择: ybpang-1

# 4. 部署并训练
modal run modal_train.py

# 5. 后台运行（笔记本可以关闭）
modal run --detach modal_train.py
```

---

## 🏗️ 模型结构

### Transformer模型

```python
class TransformerModel(nn.Module):
    """
    Transformer for sequence classification
    - 自注意力机制捕捉长程依赖
    - 位置编码表示时序信息
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads,
                 num_classes, dropout=0.1):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, seq_len, hidden_dim)
        x = self.fc(x)  # (batch, seq_len, num_classes)
        return x


class PositionalEncoding(nn.Module):
    """添加位置信息"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                            (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

### 模型参数
- Input Dim: 28
- Hidden Dim: 256
- Num Layers: 4
- Num Heads: 8
- Dropout: 0.1

---

## ⚙️ 配置文件

**文件**: `configs/config.yaml`

```yaml
# V3 Modal基础配置

# 数据设置
train_data_dir: 'data/train'
val_data_dir: 'data/val'

# 模型设置
model_type: 'transformer'
input_dim: 28
num_classes: 4
hidden_dim: 256
num_layers: 4
num_heads: 8
dropout: 0.1

# 序列设置
sequence_length: 64
frame_gap: 1

# 训练设置
epochs: 10  # 初始测试
batch_size: 16  # T4较小batch
learning_rate: 0.0001
weight_decay: 0.00001
optimizer: 'adamw'
grad_clip: 1.0

# 学习率调度
scheduler: 'cosine'
min_lr: 0.000001

# Checkpoint
checkpoint_dir: 'checkpoints'
save_freq: 5

# Device
device: 'cuda'
seed: 42
```

---

## 🚀 使用方法

### 本地开发
```bash
# 1. 安装依赖
pip install modal torch numpy pandas pyyaml

# 2. 配置Modal
modal token new
# 选择workspace: ybpang-1

# 3. 测试Modal连接
modal app list
```

### 部署训练

```bash
# 进入V3目录
cd versions/v3_modal_basic/

# 同步运行（测试用）
modal run modal_train.py

# 后台运行（推荐）
modal run --detach modal_train.py

# 查看日志
modal app logs mabe-training-basic

# 停止任务
modal app stop mabe-training-basic
```

### 下载checkpoint

```python
# download_checkpoint.py
@app.function(volumes={"/vol": volume})
def download_checkpoint(filename="best_model.pth"):
    checkpoint_path = Path("/vol/checkpoints") / filename
    with open(checkpoint_path, "rb") as f:
        return f.read()

# 本地执行
modal run download_checkpoint.py
```

---

## 📊 性能指标

### Modal性能
- **GPU**: T4 (16GB VRAM)
- **Batch Size**: 16
- **训练速度**: ~30-40 it/s
- **每epoch时间**: ~2-3分钟（小数据集）

### 相比V2的提升
- ✅ **训练速度**: 本地CPU → T4 GPU（~10x加速）
- ✅ **可扩展性**: 固定资源 → 按需伸缩
- ✅ **便捷性**: 笔记本可关闭，训练继续
- ✅ **持久化**: Volume自动保存

### 验证功能
✅ Modal部署成功
✅ T4 GPU正常工作
✅ Volume持久化正常
✅ Checkpoint保存和下载
✅ --detach后台运行

---

## 🔍 局限性

### 主要问题
1. **仍使用合成数据** - 未集成Kaggle真实数据
2. **GPU较小** - T4仅16GB，batch size受限
3. **Transformer耗资源** - 不如Conv1DBiLSTM高效
4. **训练轮数少** - 仅10 epochs，未充分训练

### 缺失功能
- ❌ 真实Kaggle竞赛数据
- ❌ 更大GPU（A10G/A100）
- ❌ Conv1DBiLSTM模型（V2中验证更好）
- ❌ Motion features
- ❌ 完整训练（100+ epochs）

---

## 💡 经验教训

### 成功点
1. ✅ Modal部署流程验证成功
2. ✅ GPU加速效果显著
3. ✅ Volume持久化可靠
4. ✅ --detach实现后台训练

### 需改进
1. ⚠️ 需要切换回Conv1DBiLSTM（V2中更好）
2. ⚠️ 需要升级GPU至A10G（24GB）
3. ⚠️ 必须集成真实Kaggle数据
4. ⚠️ 增加训练轮数至100+
5. ⚠️ 添加Motion features

### Modal使用技巧
- 💡 使用`--detach`进行长时间训练
- 💡 定期`volume.commit()`防止数据丢失
- 💡 设置合理的`timeout`参数
- 💡 使用`modal app logs`查看运行日志

---

## 📁 文件结构

```
v3_modal_basic/
├── README.md              # 本文档
├── modal_train.py         # Modal训练脚本
├── configs/
│   └── config.yaml        # Modal配置
└── docs/
    └── modal_setup.md     # Modal设置指南
```

---

## 🔄 升级到V4

### 主要改进方向
1. **模型切换** → Transformer → Conv1DBiLSTM
2. **GPU升级** → T4 → A10G (24GB)
3. **真实数据** → 准备Kaggle数据集成
4. **完整训练** → 100 epochs
5. **高级配置** → 使用V2的高级配置

### 迁移指南
```bash
# 查看V4版本（Modal高级版）
cd ../v4_modal_advanced/
cat README.md
```

---

## 📚 参考资料

### Modal文档
- [Modal Quickstart](https://modal.com/docs/guide)
- [Modal GPU Guide](https://modal.com/docs/guide/gpu)
- [Modal Volumes](https://modal.com/docs/guide/volumes)

### 代码位置
- Modal训练: `modal_train.py`
- 配置文件: `configs/config.yaml`
- Workspace: `ybpang-1`

### 相关文档
- [V2_README.md](../v2_advanced/README.md) - 上一版本
- [V4_README.md](../v4_modal_advanced/README.md) - 下一版本
- [VERSION_HISTORY.md](../../VERSION_HISTORY.md) - 完整版本历史

---

**V3 - 云端部署实现，开启GPU加速** ☁️
