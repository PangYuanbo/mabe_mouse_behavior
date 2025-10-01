# GPU 选择建议

## Modal GPU 选项对比

| GPU | VRAM | 性能 | 成本/小时 | 适用场景 |
|-----|------|------|-----------|----------|
| **T4** | 16GB | 基准 | ~$0.60 | 小模型、预算有限 |
| **A10G** ⭐ | 24GB | 3x T4 | ~$1.20 | **推荐：性价比最高** |
| **A100-40GB** | 40GB | 6x T4 | ~$4.00 | 大模型、生产环境 |
| **A100-80GB** | 80GB | 6x T4 | ~$5.00 | 超大模型 |
| **H100** | 80GB | 10x T4 | ~$8.00 | 最新、最强（通常无货）|

## 当前项目需求

### 模型规模
- **参数量**: 11.5M (Conv1DBiLSTM)
- **内存需求**: ~2GB (模型) + ~4GB (批处理) = **~6GB**
- **推荐 VRAM**: 16GB+

### 数据规模
- **训练样本**: 393 个序列
- **验证样本**: 23 个序列
- **Batch size**: 32
- **序列长度**: 100 帧 × 131 特征

### 训练时间估算

| GPU | 每个 Epoch | 100 Epochs | 成本 (100 epochs) |
|-----|-----------|------------|------------------|
| T4 | ~3分钟 | ~5小时 | ~$3.00 |
| **A10G** ⭐ | ~1分钟 | ~1.7小时 | ~$2.00 |
| A100 | ~30秒 | ~0.8小时 | ~$3.20 |

## 推荐方案

### ✅ 最佳选择: **A10G**

**理由**:
1. **性价比最高** - 速度快，价格合理
2. **VRAM 充足** - 24GB 足够当前模型和更大模型
3. **训练时间合理** - 1-2 小时完成 100 epochs
4. **可扩展性** - 支持更大 batch size 和模型

**适用于**:
- ✅ 当前 Conv1DBiLSTM 模型 (11.5M)
- ✅ Hybrid 模型实验
- ✅ 增加 batch size 到 64
- ✅ 多模型对比训练

### 备选 1: **T4** (预算紧张)

**理由**:
- 成本最低
- 16GB VRAM 勉强够用
- 训练时间可接受 (5小时)

**限制**:
- ⚠️ Batch size 需要 ≤32
- ⚠️ 不支持超大模型
- ⚠️ 速度慢 3x

### 备选 2: **A100** (土豪专用)

**理由**:
- 最快速度
- 超大 VRAM
- 适合快速迭代

**问题**:
- 💰 成本高（对当前项目浪费）
- 🎯 性能过剩（当前模型用不完）

## 不同场景推荐

### 1. 快速测试/调试
```python
gpu="T4"  # 便宜，够用
timeout=1800  # 30分钟
epochs=10  # 快速验证
```

### 2. 正式训练（推荐）⭐
```python
gpu="A10G"  # 最佳性价比
timeout=3600 * 4  # 4小时
epochs=100  # 完整训练
```

### 3. 大规模实验
```python
gpu="A100"  # 快速迭代
timeout=3600 * 2  # 2小时
epochs=200  # 充分训练
```

### 4. 超大模型（未来）
```python
gpu="A100-80GB"  # 或 H100
timeout=3600 * 8  # 8小时
# Ensemble models, Very deep networks
```

## Modal 配置示例

### 使用 A10G (推荐)
```python
@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM
    timeout=3600 * 4,  # 4小时
    volumes={"/vol": volume},
)
def train_advanced_model(...):
    # 训练代码
    pass
```

### 使用 T4 (省钱)
```python
@app.function(
    image=image,
    gpu="T4",  # 16GB VRAM
    timeout=3600 * 6,  # 6小时（因为慢）
    volumes={"/vol": volume},
)
def train_advanced_model(...):
    # 训练代码
    pass
```

### 使用 A100 (快速)
```python
@app.function(
    image=image,
    gpu="A100",  # 40GB VRAM
    timeout=3600 * 2,  # 2小时足够
    volumes={"/vol": volume},
)
def train_advanced_model(...):
    # 训练代码
    pass
```

## 性能优化建议

### 最大化 GPU 利用率

#### A10G 配置
```yaml
# config_advanced.yaml
batch_size: 64  # 可以更大
num_workers: 8  # 数据加载加速
sequence_length: 100
```

#### T4 配置
```yaml
# config_advanced.yaml
batch_size: 32  # 保持中等
num_workers: 4
sequence_length: 100
```

## 成本优化

### 1. 使用早停
```yaml
early_stopping_patience: 15  # 避免无用训练
```

### 2. 先用 T4 调试
```bash
# 调试阶段用 T4（10 epochs）
modal run modal_train_advanced.py --epochs 10

# 确认无误后用 A10G 完整训练
modal run modal_train_advanced.py --epochs 100
```

### 3. 并行训练多个模型
```python
# 同时训练 3 个模型配置
modal run modal_train_advanced.py --config config1.yaml &
modal run modal_train_advanced.py --config config2.yaml &
modal run modal_train_advanced.py --config config3.yaml &
```

## 实际成本估算

### 开发阶段（调试 + 实验）
- T4: 5小时 × $0.60 = **$3.00**
- A10G: 2小时 × $1.20 = **$2.40**

### 生产训练（完整训练）
- T4: 5小时 × $0.60 = **$3.00**
- A10G: 1.7小时 × $1.20 = **$2.04**
- A100: 0.8小时 × $4.00 = **$3.20**

### 多次迭代（5次完整训练）
- T4: 25小时 × $0.60 = **$15.00**
- A10G: 8.5小时 × $1.20 = **$10.20** ⭐ 最省
- A100: 4小时 × $4.00 = **$16.00**

## 结论

### 🎯 最佳推荐: **A10G**

**原因**:
1. ✅ 速度快（3x T4）
2. ✅ 成本低（多次训练总成本最低）
3. ✅ VRAM 充足（24GB）
4. ✅ 可扩展（支持更大模型）
5. ✅ 稳定可靠

**当前配置已是最优** - 无需更改！

```python
# modal_train_advanced.py
@app.function(
    gpu="A10G",  # ✅ 保持此配置
    ...
)
```

## 快速切换 GPU

如需测试不同 GPU，修改配置：

```python
# 选项 1: 代码中修改
gpu="T4"  # 或 "A10G" 或 "A100"

# 选项 2: 创建多个配置文件
modal_train_t4.py  # GPU="T4"
modal_train_a10g.py  # GPU="A10G"
modal_train_a100.py  # GPU="A100"
```

---

**当前使用 A10G 是最合理的选择！直接开始训练吧！** 🚀