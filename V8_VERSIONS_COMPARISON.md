# V8 版本对比总结

## 快速选择

| 使用场景 | 推荐版本 | 原因 |
|---------|---------|------|
| 最稳定训练 | V8 Safe | 保守设置，不会 NaN |
| 平衡性能 | **V8.1** | 运动门控 + 合理阈值 |
| 最高召回 | **V8.2** ⭐ | 低阈值 + kernel=3 |

**推荐**: 先用 **V8.2**，如果 sniff 类 FP 过高再降级到 V8.1

---

## 详细对比

### 核心参数对比

| 参数 | V8 Safe | V8.1 | V8.2 |
|------|---------|------|------|
| **验证平滑核** | 5 | 5 | **3** ⭐ |
| **sniffgenital** | 0.27 | 0.26 | **0.25** |
| **sniffface** | 0.28 | 0.28 | **0.26** |
| **sniffbody** | 0.30 | 0.28 | **0.25** |
| **sniffbody merge_gap** | 5 | 6 | **8** |
| **intromit** | 0.40 | 0.45 | 0.45 |
| **mount** | 0.40 | 0.43 | 0.43 |
| **approach** | 0.40 | 0.42 | 0.42 |
| **escape 速度门控** | ❌ | ✅ 80 px/s | ✅ 80 px/s |
| **freeze 速度门控** | ❌ | ✅ 7 px/s | ✅ 7 px/s |
| **Kaggle F1 跟踪** | 每 5 轮 | 每轮 | 每轮 |

---

### 功能对比

#### V8 Safe (基线)
✅ 稳定训练（无 NaN）
✅ 基础后处理
❌ 无运动门控
❌ 阈值较保守
❌ 不频繁计算 Kaggle F1

**适用**: 初次训练、寻求稳定性

---

#### V8.1 (优化版)
✅ 运动门控（escape/chase/freeze）
✅ 精调类别阈值
✅ 每轮计算 Kaggle F1
✅ 基于 Kaggle F1 保存模型
⚠️ 验证平滑核=5（可能丢失短段）

**适用**: 需要压制 escape FP、提升 freeze 召回

**改进**:
- Escape FP: -30~50%
- Freeze 召回: +10~20%
- 整体 F1: +2~5% vs V8 Safe

---

#### V8.2 (精调版) ⭐
✅ **验证平滑核=3**（更好保留短段）
✅ **更低的 sniff 阈值**（提升召回）
✅ **更大的 sniffbody 合并间隙**
✅ 保持所有 V8.1 的运动门控
✅ 每轮计算 Kaggle F1

**适用**: 需要最大化 sniff 类召回、保留短段

**改进**:
- Sniff 召回: +3~8% vs V8.1
- 短段检测: 更好
- Sniffbody 连续性: 更好
- 整体 F1: +1~3% vs V8.1, +3~6% vs V8 Safe

---

## 训练时间对比

所有版本相同:
- 每个 epoch: ~2.5 分钟
- 预计训练: 30-50 epochs
- 总时间: **1-2 小时**

**唯一区别**: V8.1/V8.2 每轮都算 Kaggle F1，验证稍慢 15-20 秒

---

## 文件结构

```
versions/
├── v8_fine_grained/           # V8 基础版本
├── v8_1_optimized_postproc/   # V8.1 后处理
└── v8_2_fine_tuned/          # V8.2 后处理 ⭐

configs/
├── config_v8_5090_safe.yaml      # V8 Safe
├── config_v8.1_optimized.yaml    # V8.1
└── config_v8.2_fine_tuned.yaml   # V8.2 ⭐

训练脚本:
├── train_v8_local.py          # V8 Safe
├── train_v8.1_local.py        # V8.1
└── train_v8.2_local.py        # V8.2 ⭐

推理脚本:
├── inference_v8.py            # V8 基础
├── inference_v8.1.py          # V8.1
└── inference_v8.2.py          # V8.2 ⭐
```

---

## 启动命令

### V8 Safe
```bash
python train_v8_local.py --config configs/config_v8_5090_safe.yaml
```

### V8.1
```bash
python train_v8.1_local.py
# 或
start_v8.1_training.bat
```

### V8.2 (推荐) ⭐
```bash
python train_v8.2_local.py
# 或
start_v8.2_training.bat
```

---

## 预期性能（验证集 F1）

估算值：

| 版本 | 整体 F1 | Escape F1 | Freeze F1 | Sniff F1 |
|------|---------|-----------|-----------|----------|
| V8 Safe | 0.630 | 0.55 | 0.45 | 0.68 |
| V8.1 | **0.655** | **0.72** | **0.58** | 0.70 |
| V8.2 | **0.670** ⭐ | **0.72** | **0.58** | **0.75** ⭐ |

*实际值取决于数据和训练*

---

## 升级路径

### 从 V8 Safe → V8.1
```bash
# 复制权重（可选）
cp checkpoints/v8_5090_safe/best_model.pth checkpoints/v8.1_optimized/pretrained.pth

# 启动训练
python train_v8.1_local.py
```

**收益**: Escape FP ↓30%, Freeze Recall ↑15%, F1 +2~5%

---

### 从 V8.1 → V8.2
```bash
# 复制权重（可选）
cp checkpoints/v8.1_optimized/best_model.pth checkpoints/v8.2_fine_tuned/pretrained.pth

# 启动训练
python train_v8.2_local.py
```

**收益**: Sniff Recall ↑5%, 短段检测更好, F1 +1~3%

---

## 调试建议

### 如果 Escape FP 过高
→ 使用 **V8.1** 或 **V8.2**（都有速度门控）

### 如果 Freeze 召回过低
→ 使用 **V8.1** 或 **V8.2**（速度 ≤7 放宽阈值）

### 如果 Sniff 召回过低
→ 使用 **V8.2**（最低阈值）

### 如果短段丢失
→ 使用 **V8.2**（kernel=3）

### 如果 Sniff FP 过高
→ 降级到 **V8.1**（稍高阈值）

---

## 最终推荐

🏆 **优先使用 V8.2**

原因:
1. 最好的短段保留（kernel=3）
2. 最高的 sniff 召回（低阈值）
3. 保持所有 V8.1 的运动门控优势
4. 每轮跟踪 Kaggle F1

**如果遇到问题再降级到 V8.1 或 V8 Safe**

---

运行命令:
```bash
start_v8.2_training.bat
```

查看详细指南:
```bash
cat V8.2_QUICK_START.md
```
