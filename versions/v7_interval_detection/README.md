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

### 训练

```bash
python train_interval_detection.py \
  --data_dir /path/to/data \
  --sequence_length 1000 \
  --batch_size 8 \
  --epochs 50 \
  --lr 1e-4
```

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
  "input_dim": 142,           # 71 keypoints × 2
  "hidden_dim": 256,
  "num_actions": 4,           # attack, avoid, chase, chaseattack
  "num_agents": 4,
  "sequence_length": 1000,
  "anchor_scales": [10, 30, 60, 120, 240],
  "iou_threshold": 0.5,
  "lr": 1e-4,
  "weight_decay": 1e-5,
  "batch_size": 8,
  "epochs": 50
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

## 下一步

- [ ] 测试训练效果
- [ ] 调优anchor scales
- [ ] 尝试不同backbone (Transformer?)
- [ ] 数据增强策略
