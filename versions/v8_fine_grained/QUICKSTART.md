# V8 Quick Start Guide

## 快速开始

### 1. 验证环境

```bash
# 检查Python和包
python --version  # 应该是 3.8+
pip list | grep torch  # 应该有 torch, torchvision

# 测试V8导入
python -c "from versions.v8_fine_grained import V8BehaviorDetector; print('V8 OK')"
```

### 2. 准备数据

确保数据在正确位置：
```
data/kaggle/
├── train_tracking/      # 关键点轨迹
├── train_annotation/    # 行为标注
├── train.csv           # 元数据
└── val.csv             # 验证集元数据
```

### 3. 训练V8模型

```bash
# 使用RTX 5090配置
python train_v8_local.py --config configs/config_v8_5090.yaml

# 或使用其他GPU（需要调整batch_size）
python train_v8_local.py --config configs/config_v8_5090.yaml --batch_size 256
```

### 4. 查看训练进度

训练过程中会显示：
```
Epoch 1/100
[Train] Action Loss: 2.34 | Agent Acc: 0.65 | Target Acc: 0.63
[Val]   Action Loss: 2.51 | Agent Acc: 0.62 | Target Acc: 0.60

[Val Metrics - Per Action Category]
  Social (7 behaviors):      F1=0.25
  Mating (4 behaviors):      F1=0.31
  Aggressive (7 behaviors):  F1=0.28
  Other (10 behaviors):      F1=0.18

[Val Interval F1] ⭐ KAGGLE SCORE ⭐
  Overall F1: 0.38
  Precision:  0.35
  Recall:     0.42
```

### 5. 生成Kaggle提交

```python
# inference.py
import torch
import pandas as pd
from versions.v8_fine_grained import V8BehaviorDetector, create_submission

# 加载模型
model = V8BehaviorDetector(input_dim=288, num_actions=28)
model.load_state_dict(torch.load('checkpoints/v8_5090/best_model.pth'))
model.eval()

# 推理
predictions = {}
for video_id, data in test_loader:
    with torch.no_grad():
        action_logits, agent_logits, target_logits = model(data)

    predictions[video_id] = {
        'action': action_logits,
        'agent': agent_logits,
        'target': target_logits
    }

# 生成提交文件
submission = create_submission(
    predictions=predictions,
    video_ids=test_video_ids,
    min_duration=5,
    confidence_threshold=0.5
)

submission.to_csv('submission_v8.csv', index=False)
print(f"Generated {len(submission)} predictions")
```

## 常见问题

### Q1: 训练很慢怎么办？
- 减小batch_size（如512→256）
- 减少num_workers（Windows建议设为0）
- 使用混合精度训练（use_amp: true）

### Q2: 显存不足？
```yaml
# configs/config_v8_5090.yaml
batch_size: 256  # 降低batch size
sequence_length: 80  # 缩短序列长度
```

### Q3: 某些行为F1很低？
这是正常的，因为：
- 稀有行为样本少（如ejaculate）
- 可以尝试：
  - 增加oversample_rare_classes
  - 调高focal_gamma（更关注困难样本）
  - 使用ensemble多模型

### Q4: 如何调优？
1. **学习率**：0.0005是起点，可以尝试0.0003-0.001
2. **Focal gamma**：2.0是标准值，稀有类多可以提高到2.5-3.0
3. **Loss权重**：action:agent:target = 1.0:0.3:0.3，可以调整
4. **数据增强**：temporal_jitter可以增加到3-5

## 性能基准

| 配置 | Interval F1 | 训练时间/epoch | 显存占用 |
|------|------------|---------------|---------|
| RTX 5090 (batch=512) | 0.38-0.42 | ~8min | ~24GB |
| RTX 3090 (batch=256) | 0.36-0.40 | ~12min | ~20GB |
| A100 (batch=768) | 0.40-0.44 | ~6min | ~30GB |

## 下一步

1. **Ensemble**: 训练多个模型并平均预测
2. **后处理**: 使用规则平滑区间边界
3. **特征工程**: 添加更多motion features
4. **层次分类**: 先预测大类，再细分

详细文档请查看：
- [V8_DESIGN.md](V8_DESIGN.md) - 设计文档
- [README.md](README.md) - 完整说明
