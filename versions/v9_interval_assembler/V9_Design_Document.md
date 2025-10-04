# V9 间隔组装器 (Interval Assembler) 设计文档

## 背景问题分析

当前V8模型存在的核心问题：
- **帧级准确率高** (Action: 84%, Agent: 98%, Target: 90%)
- **间隔级F1极低** (Interval F1: 0.2075, Precision: 0.2212, Recall: 0.1955)
- **预测过度碎片化**：模型倾向于预测背景类，真正的行为间隔被分割
- **边界定位不准**：帧级预测转换为间隔时边界误差累积
- **规则式后处理局限**：固定阈值和合并策略无法适应不同行为的时序特性

## V9设计理念

**核心思想**：引入专门的"间隔重组器"，将帧级预测智能地组装成Kaggle要求的间隔格式。

**关键优势**：
1. **端到端优化**：直接针对Interval F1优化，而非间接的帧级准确率
2. **可学习边界**：学习不同行为的最优边界定位策略
3. **智能合并**：自动学习何时合并相邻片段，何时保持分割
4. **行为自适应**：不同行为使用不同的时长和置信度要求

---

## 方案一：可学习重组器 (Learnable Assembler) 🚀

**优先级**: ⭐⭐⭐⭐⭐ (立即实施)

### 架构设计

```
输入: V8帧级logits [T, 28] + agent_logits [T, 4] + target_logits [T, 4] + 原始特征 [T, 112]
      ↓
时序编码器: TCN/BiLSTM [T, 256]
      ↓
多头输出:
├── Start边界检测: [T, 28×12] (28行为 × 12定向鼠对)
├── End边界检测:   [T, 28×12] 
├── 段内置信度:     [T, 28×12]
└── 段级特征:      [T, 128] (用于后续NMS)
```

### 关键创新点

#### 1. 定向鼠对建模
- **12个定向鼠对**: mouse1→mouse2, mouse1→mouse3, mouse1→mouse4, mouse2→mouse1, ...
- **避免agent/target混乱**: 直接为每个(action, agent, target)组合建模
- **对称性处理**: 对reciprocalsniff等双向行为特殊处理

#### 2. 软边界标签生成
```python
def generate_soft_boundary_labels(gt_intervals, sequence_length, sigma=2):
    """生成高斯平滑的边界标签，减少边界噪声"""
    start_labels = np.zeros((sequence_length, 28*12))
    end_labels = np.zeros((sequence_length, 28*12))
    
    for interval in gt_intervals:
        action_id = interval['action_id']
        pair_id = get_pair_id(interval['agent_id'], interval['target_id'])
        channel = action_id * 12 + pair_id
        
        # 高斯平滑边界
        start_frame = interval['start_frame']
        end_frame = interval['stop_frame']
        
        start_labels[:, channel] += gaussian_kernel(range(sequence_length), start_frame, sigma)
        end_labels[:, channel] += gaussian_kernel(range(sequence_length), end_frame, sigma)
    
    return start_labels, end_labels
```

#### 3. 多任务损失函数
```python
class AssemblerLoss(nn.Module):
    def forward(self, pred_start, pred_end, pred_conf, gt_start, gt_end, gt_intervals):
        # 1. 边界检测损失 (Focal Loss处理类别不平衡)
        boundary_loss = focal_loss(pred_start, gt_start) + focal_loss(pred_end, gt_end)
        
        # 2. 段级Soft-IoU损失
        pred_intervals = decode_intervals(pred_start, pred_end, pred_conf)
        soft_iou_loss = compute_soft_iou_loss(pred_intervals, gt_intervals)
        
        # 3. 反碎片化正则
        fragment_penalty = count_fragments_penalty(pred_intervals)
        
        return boundary_loss + 0.5 * soft_iou_loss + 0.1 * fragment_penalty
```

### 推理管道

#### 1. 峰值检测与配对
```python
def decode_intervals(start_heatmap, end_heatmap, confidence):
    intervals = []
    
    for action_id in range(28):
        if action_id == 0:  # 跳过背景
            continue
            
        for pair_id in range(12):
            channel = action_id * 12 + pair_id
            
            # 找峰值
            start_peaks = find_peaks(start_heatmap[:, channel], min_conf=0.3)
            end_peaks = find_peaks(end_heatmap[:, channel], min_conf=0.3)
            
            # 贪心配对
            for start_frame, start_conf in start_peaks:
                best_end = find_best_end(end_peaks, start_frame, max_duration=200)
                if best_end:
                    end_frame, end_conf = best_end
                    intervals.append({
                        'start_frame': start_frame,
                        'stop_frame': end_frame,
                        'action_id': action_id,
                        'agent_id': pair_id // 3,
                        'target_id': (pair_id % 3) + (1 if pair_id % 3 >= pair_id // 3 else 0),
                        'confidence': (start_conf + end_conf) / 2
                    })
    
    return intervals
```

#### 2. 行为自适应过滤
```python
BEHAVIOR_CONFIGS = {
    'bite': {'min_duration': 2, 'conf_threshold': 0.4},
    'flinch': {'min_duration': 2, 'conf_threshold': 0.4},
    'ejaculate': {'min_duration': 3, 'conf_threshold': 0.3},
    'sniff': {'min_duration': 5, 'conf_threshold': 0.2},
    'chase': {'min_duration': 8, 'conf_threshold': 0.3},
    'attack': {'min_duration': 4, 'conf_threshold': 0.3},
    # ... 其他行为
}

def adaptive_filter(intervals):
    filtered = []
    for interval in intervals:
        action_name = ID_TO_ACTION[interval['action_id']]
        config = BEHAVIOR_CONFIGS.get(action_name, {'min_duration': 5, 'conf_threshold': 0.3})
        
        duration = interval['stop_frame'] - interval['start_frame'] + 1
        if duration >= config['min_duration'] and interval['confidence'] >= config['conf_threshold']:
            filtered.append(interval)
    
    return filtered
```

#### 3. 时序NMS与小空洞合并
```python
def temporal_nms_and_merge(intervals, nms_iou_threshold=0.3, merge_gap_threshold=5):
    # 按置信度排序
    intervals = sorted(intervals, key=lambda x: x['confidence'], reverse=True)
    
    # NMS去重叠
    keep = []
    while intervals:
        best = intervals.pop(0)
        keep.append(best)
        
        # 移除与best重叠度高的间隔
        intervals = [itv for itv in intervals 
                    if compute_temporal_iou(best, itv) < nms_iou_threshold
                    or not same_behavior(best, itv)]
    
    # 小空洞合并
    merged = merge_nearby_intervals(keep, gap_threshold=merge_gap_threshold)
    
    return merged
```

### 训练策略

#### 数据生成
- **输入**: V8的帧级预测 + 原始关键点特征
- **标签**: 从train_annotation生成的软边界热力图 + 间隔列表
- **增强**: 时间抖动、噪声注入、片段裁剪

#### 优化目标
1. **主要目标**: 最大化Interval F1 (通过Soft-IoU逼近)
2. **辅助目标**: 边界精度、置信度校准
3. **正则化**: 反碎片化、类别平衡

---

## 方案二：Anchor式间隔检测器 (Anchor-based Interval Detector) 🎯

**优先级**: ⭐⭐⭐⭐ (第二阶段实施)

### 架构设计

```
输入: 原始特征 [T, 112] + V8帧级logits [T, 28+4+4] (可选蒸馏)
      ↓
Backbone: 1D ResNet + BiLSTM/Transformer
      ↓
Feature Pyramid: [T/1, T/2, T/4] 多尺度特征
      ↓
多尺度Anchor: 5,10,20,40,80,160,320帧 × 每个位置
      ↓
检测头:
├── Action分类: [N_anchors, 28]
├── Agent/Target分类: [N_anchors, 4+4] 
├── Objectness: [N_anchors, 1]
└── 边界回归: [N_anchors, 2] (start_offset, end_offset)
```

### 关键技术

#### 1. 多尺度Anchor策略
```python
class AnchorGenerator:
    def __init__(self, scales=[5, 10, 20, 40, 80, 160, 320], ratios=[0.5, 1.0, 2.0]):
        self.scales = scales
        self.ratios = ratios
    
    def generate_anchors(self, feature_length):
        anchors = []
        for pos in range(feature_length):
            for scale in self.scales:
                for ratio in self.ratios:
                    duration = int(scale * ratio)
                    center = pos * self.stride
                    start = center - duration // 2
                    end = center + duration // 2
                    anchors.append([start, end, duration])
        return torch.tensor(anchors)
```

#### 2. Anchor分配策略
```python
def assign_anchors(anchors, gt_intervals, pos_iou_threshold=0.5, neg_iou_threshold=0.1):
    ious = compute_temporal_iou_matrix(anchors, gt_intervals)
    
    # 正样本: IoU >= 0.5
    positive_mask = ious.max(dim=1)[0] >= pos_iou_threshold
    
    # 负样本: IoU <= 0.1  
    negative_mask = ious.max(dim=1)[0] <= neg_iou_threshold
    
    # 忽略样本: 0.1 < IoU < 0.5
    ignore_mask = ~positive_mask & ~negative_mask
    
    return positive_mask, negative_mask, ignore_mask
```

#### 3. 边界回归
```python
def encode_bbox_targets(anchors, gt_intervals):
    """将GT编码为相对anchor的偏移"""
    # 归一化偏移
    anchor_centers = (anchors[:, 0] + anchors[:, 1]) / 2
    anchor_durations = anchors[:, 1] - anchors[:, 0]
    
    gt_centers = (gt_intervals[:, 0] + gt_intervals[:, 1]) / 2
    gt_durations = gt_intervals[:, 1] - gt_intervals[:, 0]
    
    # 编码为相对偏移
    start_offsets = (gt_intervals[:, 0] - anchors[:, 0]) / anchor_durations
    end_offsets = (gt_intervals[:, 1] - anchors[:, 1]) / anchor_durations
    
    return torch.stack([start_offsets, end_offsets], dim=1)
```

### 训练损失
```python
class AnchorBasedLoss(nn.Module):
    def forward(self, predictions, targets):
        # 1. 分类损失 (Focal Loss)
        action_loss = focal_loss(pred_actions, gt_actions, alpha=0.25, gamma=2.0)
        agent_loss = focal_loss(pred_agents, gt_agents)
        target_loss = focal_loss(pred_targets, gt_targets)
        
        # 2. Objectness损失
        objectness_loss = binary_cross_entropy(pred_objectness, gt_objectness)
        
        # 3. 边界回归损失 (Smooth L1)
        regression_loss = smooth_l1_loss(pred_offsets, gt_offsets)
        
        # 4. 段级IoU损失 (只对正样本)
        decoded_intervals = decode_predictions(predictions)
        iou_loss = diou_loss(decoded_intervals, gt_intervals)
        
        return action_loss + agent_loss + target_loss + objectness_loss + regression_loss + 0.5 * iou_loss
```

---

## 方案三：结构化序列解码 (Structured Sequence Decoding) 🧠

**优先级**: ⭐⭐⭐ (第三阶段增强)

### 半监督CRF方法

#### 1. Segment-level CRF
```python
class SegmentCRF(nn.Module):
    def __init__(self, num_actions, num_pairs):
        self.num_actions = num_actions
        self.num_pairs = num_pairs
        
        # 转移矩阵: 从状态i转移到状态j的代价
        self.transition_matrix = nn.Parameter(torch.randn(num_actions * num_pairs, num_actions * num_pairs))
        
        # 持续时长先验
        self.duration_model = DurationModel(num_actions)
    
    def forward(self, emission_scores, sequence_length):
        # emission_scores: [T, num_actions * num_pairs]
        # 使用Viterbi算法找最优路径
        optimal_path = self.viterbi_decode(emission_scores, sequence_length)
        return self.path_to_intervals(optimal_path)
```

#### 2. 学习时长分布
```python
class DurationModel(nn.Module):
    def __init__(self, num_actions):
        # 为每个行为学习Gamma分布参数
        self.alpha = nn.Parameter(torch.ones(num_actions) * 2.0)
        self.beta = nn.Parameter(torch.ones(num_actions) * 0.1)
    
    def log_prob(self, duration, action_id):
        # 计算在给定行为下，特定时长的对数概率
        return torch.distributions.Gamma(self.alpha[action_id], self.beta[action_id]).log_prob(duration)
```

### 学习合并决策

```python
class MergeDecisionNetwork(nn.Module):
    def __init__(self, feature_dim=64):
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 64),  # 10维统计特征
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def extract_pair_features(self, seg1, seg2):
        """提取相邻片段对的特征"""
        features = [
            seg1['duration'], seg2['duration'],
            seg1['confidence'], seg2['confidence'],
            seg2['start_frame'] - seg1['stop_frame'],  # gap
            int(seg1['action_id'] == seg2['action_id']),  # 同行为
            int(seg1['agent_id'] == seg2['agent_id']),    # 同agent  
            int(seg1['target_id'] == seg2['target_id']),  # 同target
            abs(seg1['confidence'] - seg2['confidence']),  # 置信度差
            min(seg1['duration'], seg2['duration']) / max(seg1['duration'], seg2['duration'])  # 时长比
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def should_merge(self, seg1, seg2):
        features = self.extract_pair_features(seg1, seg2)
        merge_prob = self.feature_extractor(features)
        return merge_prob > 0.5
```

---

## 关键工程细节

### 1. 鼠对定向建模
```python
def get_pair_id(agent_id, target_id):
    """将(agent, target)映射到0-11的ID"""
    if agent_id == target_id:
        raise ValueError("Agent and target cannot be the same")
    
    # 4×3的映射 (排除对角线)
    if target_id > agent_id:
        target_id -= 1  # 调整索引
    
    return agent_id * 3 + target_id

def get_agent_target_from_pair_id(pair_id):
    """逆映射：从pair_id恢复(agent_id, target_id)"""
    agent_id = pair_id // 3
    target_offset = pair_id % 3
    target_id = target_offset if target_offset < agent_id else target_offset + 1
    return agent_id, target_id
```

### 2. 边界鲁棒性增强
```python
def gaussian_boundary_smoothing(labels, sigma=2):
    """对边界标签进行高斯平滑，增强鲁棒性"""
    smoothed = np.zeros_like(labels)
    kernel_size = 6 * sigma + 1
    kernel = np.exp(-0.5 * ((np.arange(kernel_size) - 3*sigma) / sigma)**2)
    kernel = kernel / kernel.sum()
    
    for i in range(labels.shape[1]):
        smoothed[:, i] = np.convolve(labels[:, i], kernel, mode='same')
    
    return smoothed

def soft_iou_loss(pred_intervals, gt_intervals, sigma=5.0):
    """可微分的Soft-IoU损失，逼近0.5阈值效果"""
    total_loss = 0
    for pred in pred_intervals:
        best_iou = 0
        for gt in gt_intervals:
            if pred['action_id'] == gt['action_id'] and \
               pred['agent_id'] == gt['agent_id'] and \
               pred['target_id'] == gt['target_id']:
                # 计算soft IoU
                intersection = torch.min(pred['end'], gt['end']) - torch.max(pred['start'], gt['start'])
                intersection = torch.clamp(intersection, min=0)
                union = (pred['end'] - pred['start']) + (gt['end'] - gt['start']) - intersection
                soft_iou = intersection / (union + 1e-6)
                best_iou = torch.max(best_iou, soft_iou)
        
        # 鼓励IoU接近0.5以上
        total_loss += torch.clamp(0.5 - best_iou, min=0)**2
    
    return total_loss / len(pred_intervals)
```

### 3. 自适应阈值学习
```python
class AdaptiveThresholdModule(nn.Module):
    def __init__(self, num_actions):
        super().__init__()
        # 为每个行为学习最优阈值
        self.min_duration_logits = nn.Parameter(torch.log(torch.tensor([5.0] * num_actions)))
        self.conf_threshold_logits = nn.Parameter(torch.log(torch.tensor([0.3] * num_actions)))
        self.nms_threshold_logits = nn.Parameter(torch.log(torch.tensor([0.3] * num_actions)))
    
    def get_thresholds(self, action_id):
        return {
            'min_duration': torch.exp(self.min_duration_logits[action_id]),
            'conf_threshold': torch.sigmoid(self.conf_threshold_logits[action_id]),
            'nms_threshold': torch.sigmoid(self.nms_threshold_logits[action_id])
        }
```

---

## 落地实施计划

### 第1周：V9-Assembler原型 🚀
- [ ] 创建训练数据生成管道（软边界标签）
- [ ] 实现TCN/BiLSTM重组器
- [ ] 训练边界检测 + Soft-IoU多任务损失
- [ ] 实现自适应解码与NMS
- [ ] 在验证集上评估Interval F1提升

### 第2-3周：V9-Proposal完整版 🎯  
- [ ] 引入多尺度anchor和FPN特征
- [ ] 加入成对关系特征（距离、速度、朝向）
- [ ] 实现anchor分配和边界回归
- [ ] 蒸馏V8帧级知识
- [ ] 完善解码管道和阈值自适应

### 第4周：结构化解码增强 🧠
- [ ] 实现segment-level CRF
- [ ] 训练合并决策网络
- [ ] 集成时长先验模型
- [ ] 端到端微调所有模块

### 预期效果
- **第1周后**: Interval F1 从 0.21 → 0.45+
- **第3周后**: Interval F1 达到 0.60+
- **第4周后**: Interval F1 达到 0.70+，接近SOTA水平

---

## 总结

V9的核心理念是**"帧级准确率已经够好，关键在于如何智能组装"**。通过引入专门的间隔重组器，我们可以：

1. **直接优化目标指标**：不再依赖帧级准确率的间接提升
2. **学习行为时序特性**：不同行为的边界、时长、置信度要求
3. **端到端可微分**：从特征到最终Interval F1的完整梯度流
4. **工程化友好**：分阶段实施，风险可控，快速迭代

这个设计既保留了V8的强大帧级能力，又解决了间隔转换的核心痛点，是当前最有希望大幅提升Kaggle得分的方案。