# V9 间隔组装器 (Interval Assembler)

## 项目概览

V9是专门用于解决"帧级准确率高但间隔级F1低"问题的新架构。它不再依赖简单的规则后处理，而是引入专门的神经网络来学习如何将帧级预测智能地组装成高质量的行为间隔。

## 核心设计理念

**问题**：V8模型帧级准确率84%+，但间隔F1只有0.21
**根因**：规则式后处理无法适应不同行为的时序特性
**解决**：引入可学习的"间隔重组器"，端到端优化Interval F1

## 三个渐进方案

### 🚀 方案一：可学习重组器 (优先级★★★★★)
- **核心**：在V8基础上加边界检测头
- **输入**：V8帧级logits + 原始特征
- **输出**：Start/End边界热力图 + 置信度
- **优势**：最小改动，快速见效

### 🎯 方案二：Anchor式检测器 (优先级★★★★)
- **核心**：多尺度anchor直接生成间隔
- **架构**：FPN + 检测头 (action/agent/target/边界回归)
- **优势**：端到端，边界更准确

### 🧠 方案三：结构化解码 (优先级★★★)
- **核心**：CRF/时长模型进一步优化
- **技术**：Segment-level CRF + 合并决策网络
- **优势**：最大化结构化约束

## 预期效果

| 阶段 | 方案 | 预期Interval F1 | 开发周期 |
|------|------|----------------|----------|
| 当前 | V8规则后处理 | 0.21 | - |
| 第1周 | V9-Assembler | 0.45+ | 1周 |
| 第3周 | V9-Proposal | 0.60+ | 2-3周 |
| 第4周 | V9-Structured | 0.70+ | 4周 |

## 关键技术突破

1. **定向鼠对建模**：12个(agent→target)组合，避免混乱
2. **软边界标签**：高斯平滑减少边界噪声
3. **Soft-IoU损失**：直接优化Kaggle评估指标
4. **行为自适应阈值**：bite用2帧，sniff用5帧，chase用8帧
5. **智能NMS**：学习何时合并，何时分割

## 文件结构规划

```
versions/v9_interval_assembler/
├── README.md                    # 项目说明
├── V9_Design_Document.md        # 详细设计文档
├── __init__.py
├── models/
│   ├── __init__.py
│   ├── assembler.py             # 方案一：重组器模型
│   ├── anchor_detector.py       # 方案二：anchor检测器
│   └── structured_decoder.py    # 方案三：结构化解码器
├── data/
│   ├── __init__.py
│   ├── boundary_labels.py       # 边界标签生成
│   ├── pair_mapping.py          # 鼠对ID映射
│   └── dataset.py               # V9训练数据集
├── losses/
│   ├── __init__.py
│   ├── soft_iou.py              # Soft-IoU损失
│   ├── focal_loss.py            # Focal Loss
│   └── assembler_loss.py        # 多任务损失
├── utils/
│   ├── __init__.py
│   ├── interval_ops.py          # 间隔操作工具
│   ├── nms.py                   # 时序NMS
│   └── metrics.py               # 评估指标
├── inference/
│   ├── __init__.py
│   ├── decoder.py               # 解码器
│   └── post_process.py          # 后处理
└── train/
    ├── __init__.py
    ├── train_assembler.py       # 训练脚本
    └── config.py                # 训练配置
```

## 立即行动计划

1. **今天**：完成方案一的基础架构代码
2. **明天**：实现边界标签生成和数据管道
3. **后天**：训练第一个可学习重组器模型
4. **本周末**：在验证集上评估F1提升

## 与现有代码集成

- **复用V8**：继续使用V8的帧级预测作为输入
- **兼容训练**：使用现有的train_annotation数据
- **渐进替换**：逐步替换inference脚本中的后处理部分

V9的成功将标志着从"规则工程"到"学习工程"的转变，这是解决当前瓶颈的关键突破！