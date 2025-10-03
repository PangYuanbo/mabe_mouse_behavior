# V7: Temporal Action Detection (Interval Detection)

## Overview

V7 采用**时序动作检测 (Temporal Action Detection)** 方法，直接预测行为区间，而不是逐帧分类。

### 与之前版本的主要区别

**V5 (逐帧分类):**
- 输出：每帧的行为类别
- 损失：Cross-entropy loss
- 后处理：合并连续帧为区间

**V7 (区间检测):**
- 输出：行为区间 (start_frame, end_frame, action, agent, target)
- 损失：IoU loss + classification loss
- 直接优化区间预测，与比赛格式一致

## 架构

### 1. 数据加载器 (`interval_dataset.py`)
- 输出区间格式标注
- 返回：(sequence, intervals)
- intervals: [(start, end, action_id, agent_id, target_id), ...]

### 2. 模型 (`interval_model.py`)
**TemporalActionDetector**
- Feature extraction: Conv1D + BiLSTM
- Anchor-based detection (类似目标检测)
- 多个检测头：
  - Action classification
  - Agent/Target classification
  - Boundary regression (start/end offsets)
  - Objectness score (foreground/background)

### 3. 损失函数 (`interval_loss.py`)
**IntervalDetectionLoss**
- IoU loss: 优化区间边界
- Classification loss: action, agent, target
- Focal loss: objectness (处理不平衡)
- Anchor matching: 基于IoU分配GT

### 4. 评估指标 (`interval_metrics.py`)
**IntervalMetrics**
- IoU-based matching
- F1 score (与比赛评分一致)
- Per-action metrics

## 使用方法

### 快速开始

#### 1. 测试Motion Features
```bash
# 从项目根目录运行
python test_v7_motion_features.py
```

预期输出:
```
✓ Motion features shape correct!
✓ Original coordinates preserved
✓ All tests passed!
```

#### 2. 内存估算 (可选)
```bash
python estimate_v7_memory.py
```

#### 3. 训练 (标准配置)
```bash
python train_v7_local.py --config configs/config_v7_5090.yaml
```

或使用最大化配置:
```bash
python train_v7_local.py --config configs/config_v7_5090_max.yaml
```

#### 4. 一键启动 (Linux/Mac)
```bash
chmod +x run_v7_training.sh
./run_v7_training.sh
```

### 详细训练命令

```bash
# 标准配置 (推荐首次使用)
python train_v7_local.py \
  --config configs/config_v7_5090.yaml

# 最大化性能
python train_v7_local.py \
  --config configs/config_v7_5090_max.yaml
```

### 配置说明

**标准配置** (`config_v7_5090.yaml`):
- Batch size: 32
- Mixed precision: enabled
- Motion features: enabled
- 预估时间: ~10小时 (100 epochs)

**最大化配置** (`config_v7_5090_max.yaml`):
- Batch size: 48 (激进)
- 充分利用RTX 5090 32GB VRAM
- 预估时间: ~7小时 (100 epochs)

### 推理

```python
from interval_model import TemporalActionDetector
import torch

# Load model
model = TemporalActionDetector(input_dim=142, ...)
checkpoint = torch.load('best_model_v7.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict
with torch.no_grad():
    predictions = model(sequence)
    intervals = model.predict_intervals(
        predictions,
        score_threshold=0.5,
        nms_threshold=0.3
    )

# Output format
for interval in intervals[0]:
    print(interval)
    # {'start_frame': 10, 'end_frame': 50,
    #  'action_id': 2, 'agent_id': 1, 'target_id': 2, 'score': 0.85}
```

## 关键特性

### 1. Anchor-based Detection
- 多尺度anchors: [10, 30, 60, 120, 240] 帧
- 每个位置预测多个scale的区间
- 类似于Faster R-CNN的RPN

### 2. IoU Loss
- 直接优化IoU而不是L1/L2距离
- 更好地匹配评估指标

### 3. Focal Loss
- 处理正负样本不平衡
- alpha=0.25, gamma=2.0

### 4. NMS (Non-Maximum Suppression)
- 去除重复检测
- IoU threshold = 0.3

## 评估指标

比赛评分与V7评估一致：
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1: 2 * P * R / (P + R)

匹配标准：
- IoU >= 0.5
- action, agent, target 完全匹配

## 训练配置

```python
{
  "input_dim": 284,           # 71 keypoints × 4 (x, y, speed, accel)
                              # 142 (coords) + 71 (speed) + 71 (accel)
  "hidden_dim": 256,
  "num_actions": 4,           # attack, avoid, chase, chaseattack
  "num_agents": 4,
  "sequence_length": 1000,
  "anchor_scales": [10, 30, 60, 120, 240],
  "iou_threshold": 0.5,
  "use_motion_features": true,  # Enable speed & acceleration
  "fps": 33.3,                  # For motion computation
  "lr": 1e-4,
  "weight_decay": 1e-5,
  "batch_size": 16,
  "epochs": 100
}
```

## 输出格式

直接生成比赛要求的格式：

```csv
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
0,438887472,1,2,chase,10,54
1,438887472,1,3,attack,128,234
...
```

## 优势

1. **直接优化目标**：损失函数与评估指标一致
2. **端到端训练**：无需后处理启发式规则
3. **更好的边界**：IoU loss比分类后合并更准确
4. **多尺度检测**：自适应不同长度的行为
5. **Motion Features** ⭐：速度+加速度特征增强行为区分度
   - 速度捕捉运动趋势 (chase vs avoid)
   - 加速度捕捉动作变化 (attack突发性)

## Motion Features 说明

### 计算方式
```python
# 输入: [T, 142] (71 keypoints × 2 coords)

# 1. 速度 (Velocity)
velocity[t] = (position[t] - position[t-1]) / dt
speed[t] = ||velocity[t]||  # 向量长度 [T, 71]

# 2. 加速度 (Acceleration)
acceleration[t] = (velocity[t] - velocity[t-1]) / dt
accel[t] = ||acceleration[t]||  # 向量长度 [T, 71]

# 输出: [T, 284] = [142 coords + 71 speed + 71 accel]
```

### 为什么有效
- **Chase**: 高速度，低加速度（持续追逐）
- **Attack**: 高加速度（突发动作）
- **Avoid**: 速度方向与agent相反
- **Background**: 低速度，低加速度

### V6 vs V7 (都使用Motion Features)
| 特性 | V6 | V7 |
|------|----|----|
| 任务 | 逐帧分类 | 区间检测 |
| 输入 | 288维 (144+72+72) | 284维 (142+71+71) |
| 序列长度 | 100帧 | 1000帧 |
| 优化目标 | 帧准确率 | 区间IoU |

## 下一步

- [x] 添加Motion Features支持
- [ ] 测试训练效果
- [ ] 对比无Motion vs 有Motion
- [ ] 调优anchor scales
- [ ] 尝试不同backbone (Transformer?)
- [ ] 数据增强策略
