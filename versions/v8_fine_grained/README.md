# V8 Fine-grained Behavior Detection

## 概述

V8是完全符合Kaggle MABe竞赛要求的细粒度行为检测系统，相比V7的主要改进：

| 特性 | V7 | V8 |
|------|----|----|
| 行为类别 | 3类（粗粒度） | **28类（细粒度）** |
| Agent/Target | ❌ 不预测 | ✅ 预测哪两只老鼠 |
| 提交格式 | 不完整 | ✅ 完整Kaggle格式 |
| 模型架构 | 单任务 | **多任务学习** |

## 核心特性

### 1. 细粒度行为分类（28类）

```python
Social (7):      sniff, sniffgenital, sniffface, sniffbody,
                 reciprocalsniff, approach, follow

Mating (4):      mount, intromit, attemptmount, ejaculate

Aggressive (7):  attack, chase, chaseattack, bite,
                 dominance, defend, flinch

Other (10):      avoid, escape, freeze, allogroom, shepherd,
                 disengage, run, dominancegroom, huddle
```

### 2. 多任务学习架构

```
Input [B,T,288] (keypoints + motion)
    ↓
Conv1D Backbone (128→256→512)
    ↓
BiLSTM (256 hidden × 2 layers)
    ↓
    ├─→ Action Head [B,T,28]   (行为分类)
    ├─→ Agent Head [B,T,4]     (哪只鼠是agent)
    └─→ Target Head [B,T,4]    (哪只鼠是target)
```

### 3. 损失函数

```python
Total Loss = α * Action Loss + β * Agent Loss + γ * Target Loss

# 默认权重
α = 1.0   # 行为分类最重要
β = 0.3   # Agent次要
γ = 0.3   # Target次要

# Action Loss使用Focal Loss处理类别不平衡
FL(p_t) = -(1 - p_t)^γ * log(p_t)  (γ=2.0)
```

### 4. Kaggle提交格式

```csv
row_id,video_id,agent_id,target_id,action,start_frame,stop_frame
0,438887472,mouse1,mouse2,sniff,147,361
1,438887472,mouse1,mouse2,sniffgenital,362,432
...
```

## 文件结构

```
versions/v8_fine_grained/
├── README.md                    # 本文件
├── V8_DESIGN.md                 # 详细设计文档
├── action_mapping.py            # 28类行为映射
├── v8_model.py                  # 多任务模型
├── v8_dataset.py                # 数据集（带agent/target标签）
├── submission_utils.py          # Kaggle提交格式转换
├── advanced_postprocessing.py   # 高级后处理
├── train_v8.py                  # 训练脚本
├── inference_v8.py              # 推理脚本
├── inference_v8_improved.py     # 改进推理脚本
├── test_v8.py                   # 组件测试
└── test_postprocessing.py       # 后处理测试
```

## 快速开始

### 1. 训练

```bash
# RTX 5090 (32GB VRAM)
python versions/v8_fine_grained/train_v8.py --config configs/config_v8_5090.yaml

# 关键配置
batch_size: 512        # 5090优化
num_actions: 28        # 细粒度分类
use_focal_loss: true   # 处理类别不平衡
```

### 2. 生成提交文件

```python
from versions.v8_fine_grained.submission_utils import create_submission

# 模型推理
action_logits, agent_logits, target_logits = model(test_data)

# 转换为Kaggle格式
submission_df = create_submission(
    action_logits=action_logits,
    agent_logits=agent_logits,
    target_logits=target_logits,
    video_ids=test_video_ids,
    min_duration=5  # 最小区间长度
)

# 保存
submission_df.to_csv('submission.csv', index=False)
```

## 关键改进

### vs V7

1. **细粒度分类**：
   - V7: 3个大类 → V8: 28个具体行为
   - 完全符合Kaggle评分标准

2. **Agent/Target预测**：
   - V7: 只预测行为 → V8: 同时预测哪两只老鼠
   - 提交格式完整

3. **类别不平衡**：
   - V7: 简单加权 → V8: Focal Loss + 分层采样
   - 更好处理稀有行为

4. **多任务学习**：
   - V7: 单任务 → V8: 3个任务联合训练
   - 共享表征提升性能

## 预期性能

| 指标 | V7 | V8（初版） | V8（优化后） |
|------|----|-----------| -------------|
| Frame Acc | 72% | 60-65% | 65-70% |
| Interval F1 | 0.32 | 0.35-0.40 | **0.45-0.55** |
| 训练时间/epoch | 5min | 8min | 8min |

注：V8性能可能初期略低（类别更多），但优化后应该显著提升。

## 已知挑战

1. **数据稀疏**：
   - 一些行为（如ejaculate）样本极少
   - 解决：过采样 + Focal Loss

2. **Agent/Target难度高**：
   - 需要理解空间关系
   - 解决：使用关键点相对位置特征

3. **计算成本**：
   - 28类比3类计算量更大
   - 解决：混合精度训练 + 批量优化

## 后续优化方向

1. **模型ensemble**：多个V8模型投票
2. **后处理优化**：区间平滑、时序一致性
3. **数据增强**：时间扭曲、空间抖动
4. **层次化分类**：先预测大类，再细分
5. **自监督预训练**：在大量无标注数据上预训练

## 参考

- Kaggle Competition: https://www.kaggle.com/c/mabe-2022
- V7 Documentation: ../v7_interval_detection/
- MABe Challenge: http://www.mousebehavior.org/
