# V8 Fine-grained Behavior Detection - Design Document

## 目标
完全符合Kaggle MABe竞赛要求的细粒度行为检测系统

## 核心改进

### 1. 细粒度行为分类（30+类别）

**标签映射策略**：
```python
ACTION_MAPPING = {
    # Background
    'background': 0,

    # Social investigation (1-10)
    'sniff': 1,
    'sniffgenital': 2,
    'sniffface': 3,
    'sniffbody': 4,
    'reciprocalsniff': 5,
    'approach': 6,
    'follow': 7,

    # Mating behaviors (11-14)
    'mount': 11,
    'intromit': 12,
    'attemptmount': 13,
    'ejaculate': 14,

    # Aggressive behaviors (15-21)
    'attack': 15,
    'chase': 16,
    'chaseattack': 17,
    'bite': 18,
    'dominance': 19,
    'defend': 20,
    'flinch': 21,

    # Other behaviors (as background or separate classes)
    'rear': 0,
    'avoid': 22,
    'escape': 23,
    'freeze': 24,
    'selfgroom': 0,
    'allogroom': 25,
    'rest': 0,
    'dig': 0,
    'climb': 0,
    'shepherd': 26,
    'disengage': 27,
    'run': 28,
    'exploreobject': 0,
    'biteobject': 0,
    'dominancegroom': 29,
    'huddle': 30,
    'other': 0,
}

NUM_CLASSES = 31  # 0-30
```

### 2. 多任务学习架构

```
Input: [B, T, 288] (keypoints + motion features)
    ↓
Shared Backbone (Conv1D + BiLSTM)
    ↓
    ├─→ Behavior Classification Head [B, T, 31]
    │   (预测行为类别)
    │
    ├─→ Agent Identification Head [B, T, 4]
    │   (预测哪只老鼠是agent)
    │
    └─→ Target Identification Head [B, T, 4]
        (预测哪只老鼠是target)
```

### 3. 数据组织

**当前问题**：
- 数据标注格式：`(agent_id, target_id, action, start_frame, stop_frame)`
- 但输入是4只老鼠的混合关键点序列

**解决方案**：
- **保持帧级别预测**，每帧输出：
  - 行为类别 (31类)
  - Agent ID (0-3，表示mouse1-4)
  - Target ID (0-3，表示mouse1-4)
- 后处理时合并连续帧为区间

### 4. 损失函数

```python
total_loss = (
    α * action_loss +          # CrossEntropy for 31 classes
    β * agent_loss +           # CrossEntropy for 4 agents
    γ * target_loss            # CrossEntropy for 4 targets
)

# 建议权重
α = 1.0  # 行为分类最重要
β = 0.3  # Agent次要
γ = 0.3  # Target次要
```

### 5. 类别不平衡处理

**问题**：
- Social行为（sniff系列）占比>70%
- Mating/Aggressive占比<10%
- 30个类别严重不平衡

**策略**：
1. **分层采样**：
   - 按行为类别分层
   - 稀有类别过采样

2. **Focal Loss**：
   ```python
   FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
   ```
   - 自动降低简单样本权重
   - 聚焦困难样本

3. **类别权重**：
   ```python
   class_weights[i] = total_samples / (num_classes * count[i])
   ```

### 6. 提交格式转换

```python
def predictions_to_submission(
    frame_preds: np.ndarray,      # [T, 31] action probabilities
    agent_preds: np.ndarray,       # [T, 4] agent probabilities
    target_preds: np.ndarray,      # [T, 4] target probabilities
    video_id: str,
    min_duration: int = 5
) -> pd.DataFrame:
    """
    Convert frame predictions to Kaggle submission format

    Returns:
        DataFrame with columns:
        [row_id, video_id, agent_id, target_id, action, start_frame, stop_frame]
    """
    # 1. Get predicted classes
    action_classes = np.argmax(frame_preds, axis=1)
    agent_classes = np.argmax(agent_preds, axis=1)
    target_classes = np.argmax(target_preds, axis=1)

    # 2. Merge consecutive frames into intervals
    intervals = []
    current_action = None
    current_agent = None
    current_target = None
    start_frame = 0

    for t in range(len(action_classes)):
        action = action_classes[t]
        agent = agent_classes[t]
        target = target_classes[t]

        # Skip background
        if action == 0:
            if current_action is not None:
                # End current interval
                if t - start_frame >= min_duration:
                    intervals.append({
                        'agent_id': f'mouse{current_agent + 1}',
                        'target_id': f'mouse{current_target + 1}',
                        'action': ID_TO_ACTION[current_action],
                        'start_frame': start_frame,
                        'stop_frame': t - 1
                    })
                current_action = None
            continue

        # Check if same interval continues
        if (action == current_action and
            agent == current_agent and
            target == current_target):
            continue
        else:
            # Save previous interval
            if current_action is not None and t - start_frame >= min_duration:
                intervals.append({
                    'agent_id': f'mouse{current_agent + 1}',
                    'target_id': f'mouse{current_target + 1}',
                    'action': ID_TO_ACTION[current_action],
                    'start_frame': start_frame,
                    'stop_frame': t - 1
                })

            # Start new interval
            current_action = action
            current_agent = agent
            current_target = target
            start_frame = t

    # Create submission dataframe
    df = pd.DataFrame(intervals)
    df['video_id'] = video_id
    df['row_id'] = range(len(df))

    return df[['row_id', 'video_id', 'agent_id', 'target_id',
               'action', 'start_frame', 'stop_frame']]
```

### 7. 评估指标

保持双重评估：
- **Frame-level Accuracy**：快速监控
- **Interval-level F1**：Kaggle官方指标
  - 需要匹配：(action, agent, target, IoU>=0.5)

### 8. 模型架构细节

```python
class V8BehaviorDetector(nn.Module):
    def __init__(self, input_dim=288, num_actions=31):
        super().__init__()

        # Shared backbone
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )

        # Task-specific heads
        self.action_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_actions)
        )

        self.agent_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # 4 mice
        )

        self.target_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 4)  # 4 mice
        )

    def forward(self, x):
        # x: [B, T, D]
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.conv_layers(x)  # [B, 512, T]
        x = x.transpose(1, 2)  # [B, T, 512]

        x, _ = self.lstm(x)  # [B, T, 512]

        action_logits = self.action_head(x)  # [B, T, 31]
        agent_logits = self.agent_head(x)    # [B, T, 4]
        target_logits = self.target_head(x)  # [B, T, 4]

        return action_logits, agent_logits, target_logits
```

## 实现计划

### Phase 1: 数据准备
- [ ] 更新action_mapping到31类
- [ ] 添加agent/target标签提取
- [ ] 实现分层采样

### Phase 2: 模型实现
- [ ] V8模型架构
- [ ] 多任务损失函数
- [ ] Focal Loss实现

### Phase 3: 训练
- [ ] 训练脚本
- [ ] 双重评估指标
- [ ] 超参数调优

### Phase 4: 推理和提交
- [ ] 区间转换逻辑
- [ ] Kaggle提交格式
- [ ] 后处理优化

## 预期性能

- **Frame Accuracy**: 60-70%
- **Interval F1**: 0.35-0.45（初版）
- **优化后**: 0.50+（加入ensemble、后处理等）

## 关键挑战

1. **数据稀疏**：30+类别，很多行为样本极少
2. **Agent/Target预测难度**：需要理解老鼠之间的空间关系
3. **计算成本**：更多类别需要更大模型
4. **类别不平衡**：需要精心设计采样和损失函数
