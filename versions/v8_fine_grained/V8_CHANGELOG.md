# V8 Fine-Grained Behavior Detection - Changelog

## V8 vs V7 主要改进

### 🎯 核心改进

**V7问题**: 只预测3个粗粒度类别(social/mating/aggressive)，不符合Kaggle要求

**V8解决方案**: 完整实现Kaggle竞赛格式的28类精细行为检测

### 📊 关键差异对比

| 特性 | V7 | V8 |
|------|----|----|
| **行为类别** | 3类粗粒度 | 28类精细粒度 |
| **任务类型** | 单任务(仅动作) | 多任务(动作+施动者+受动者) |
| **输出格式** | 区间检测 | Kaggle标准格式 |
| **Agent/Target预测** | ❌ 不支持 | ✅ 完整支持 |
| **Kaggle兼容性** | ❌ 不兼容 | ✅ 完全兼容 |

### 🆕 V8新增功能

#### 1. 28类精细行为分类
```python
行为类别映射:
- 社交行为 (7种): sniff, sniff_genital, sniff_face, approach, follow, etc.
- 交配行为 (4种): mount, intromit, attempt_mount, ejaculate
- 攻击行为 (7种): attack, chase, chase_attack, bite, dominance, etc.
- 其他行为 (10种): avoid, escape, freeze, allogroom, etc.
```

#### 2. 多任务学习架构
```python
class V8BehaviorDetector:
    - 共享CNN+LSTM骨干网络
    - action_head: [B,T] → 28类动作
    - agent_head: [B,T] → 4只老鼠(施动者)
    - target_head: [B,T] → 4只老鼠(受动者)
```

#### 3. Focal Loss处理类别不平衡
```python
FL(p_t) = -(1 - p_t)^γ * log(p_t)
gamma=2.0: 聚焦困难样本
```

#### 4. Kaggle提交格式
```python
输出格式:
row_id, video_id, agent_id, target_id, action, start_frame, stop_frame
0, video1, mouse1, mouse2, sniff, 50, 120
1, video1, mouse2, mouse3, mount, 200, 350
```

### 🐛 已修复问题

#### 问题1: NaN Loss
**原因**: 混合精度训练(FP16)导致Focal Loss数值不稳定
**解决**: 禁用混合精度 `use_amp: false`
**结果**: Loss正常下降 3.9 → 0.4, 准确率81%

#### 问题2: 变长关键点维度
**原因**: 不同视频有不同的bodypart数量
**解决**: 标准化为7个固定身体部位 × 4只老鼠 = 112维
```python
标准身体部位: nose, leftear, rightear, neck, lefthip, righthip, tail
```

#### 问题3: Float类型错误
**原因**: `num_frames = df['video_frame'].max() + 1` 返回float
**解决**: 强制转换 `num_frames = int(df['video_frame'].max()) + 1`

### 📈 训练性能

**RTX 5090配置**:
- Batch Size: 512
- 序列长度: 100帧
- 学习率: 0.0001 (降低以防NaN)
- 显存占用: ~24GB / 32GB
- 训练速度: ~18 it/s

**当前训练指标** (Epoch 1):
```
Action准确率: 6% → 81% (快速上升)
Agent准确率: 98% (稳定)
Target准确率: ~98% (预计)
Loss: 3.9 → 0.4 (正常下降)
```

### 🔧 配置建议

**如果遇到显存不足**:
```yaml
batch_size: 256  # 降低batch size
sequence_length: 80  # 缩短序列
```

**如果训练不稳定**:
```yaml
learning_rate: 0.00005  # 进一步降低lr
focal_gamma: 1.5  # 减小gamma值
```

**如果某些类别F1很低**:
- 正常现象(稀有行为样本少)
- 可尝试oversample_rare_classes: true
- 使用ensemble多模型投票

### 📁 文件结构

```
versions/v8_fine_grained/
├── __init__.py              # 包导出
├── action_mapping.py        # 28类行为映射
├── v8_model.py             # 多任务模型 + Focal Loss
├── v8_dataset.py           # 数据集(带agent/target)
├── submission_utils.py      # Kaggle提交格式转换
├── V8_DESIGN.md            # 设计文档
├── V8_CHANGELOG.md         # 本文档
└── QUICKSTART.md           # 快速开始指南

configs/
└── config_v8_5090.yaml     # RTX 5090配置

train_v8_local.py           # 训练脚本
test_v8.py                  # 测试脚本
```

### 🚀 快速开始

```bash
# 1. 测试所有组件
python test_v8.py

# 2. 开始训练
python train_v8_local.py --config configs/config_v8_5090.yaml

# 3. 生成Kaggle提交 (训练完成后)
python inference_v8.py --checkpoint checkpoints/v8_5090/best_model.pth
```

### ✅ V8测试验证

所有组件测试通过:
- ✅ 动作映射 (28类)
- ✅ 模型前向传播
- ✅ 多任务损失计算
- ✅ 区间转换
- ✅ Kaggle提交格式生成

### 🎯 下一步优化方向

1. **Ensemble**: 训练多个模型平均预测
2. **后处理**: 平滑区间边界、去除短区间
3. **特征工程**: 添加相对位置、距离等特征
4. **层次分类**: 先预测大类，再细分
5. **类别权重优化**: 根据验证集F1调整loss权重

---

**版本**: V8.0
**日期**: 2025-10-03
**状态**: ✅ 训练中，指标正常
