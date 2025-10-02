# V6 Training Summary - Quick Reference

## 🎯 一句话总结

**V6训练完成：F1 Macro 0.6164，相比V5提升42%，Motion Features证明关键作用。**

---

## 📊 核心指标

```
Best Model: Epoch 18
F1 Macro:   0.6164 ⭐
Accuracy:   0.6821
Loss:       0.9837

Training:   33 epochs, 6 hours, A10G GPU
Dataset:    863 videos (708 train / 155 val)
```

---

## 🏆 Per-Class Performance

| Class | F1 | Precision | Recall | 评价 |
|-------|--------|-----------|--------|------|
| Background | 0.77 | 0.92 ✅ | 0.66 | 高精度但召回偏低 |
| Social | 0.51 | 0.38 ⚠️ | 0.79 | 过度预测问题 |
| Mating | 0.70 ✅ | 0.72 | 0.68 | 最佳表现 |
| Aggressive | 0.49 | 0.43 | 0.56 | 需要改进 |

---

## 📈 V5 vs V6

| Metric | V5 | V6 | Gain |
|--------|----|----|------|
| F1 Macro | 0.43 | 0.62 | **+42%** |
| Input Dim | 144 | 288 | +100% |
| Features | Coords only | + Speed + Accel | NEW |
| Training Time | 12h | 6h | 2x faster |

---

## ✅ What Worked

1. **Motion Features** - +42% F1提升的唯一原因
2. **Large Batch (384)** - 训练稳定快速
3. **A10G GPU** - 成本效益最优
4. **Early Stopping** - Best model在Epoch 18

---

## ⚠️ Known Issues

1. **Social过度预测** - Predicted 38.5% vs Actual 18.4%
2. **Background召回低** - 仅66%，34%被误分类
3. **Aggressive F1低** - 仅0.49，与其他类混淆

---

## 🚀 Next Steps

### Immediate (Today)

1. ✅ **Kaggle提交** - 使用best_model.pth (Epoch 18)
   ```bash
   python generate_submission.py --checkpoint best_model.pth
   ```

### Short-term (本周)

2. **Fix Social过度预测**
   - 尝试Focal Loss
   - 调整class weights: [1.0, 8.0, 10.0, 10.0]

3. **Test Time Augmentation**
   - 水平翻转
   - 时序增强

### Long-term (V7)

4. **Architecture升级**
   - Transformer encoder
   - Multi-head attention
   - Expected: +5-10% F1

5. **Ensemble**
   - 多个checkpoint组合
   - Expected: +3-5% F1

---

## 📁 Key Files

```
✅ Production Model:
   checkpoints/v6_a10g/best_model.pth

📊 Complete Report:
   V6_TRAINING_REPORT.md

📜 Training History:
   checkpoints/v6_a10g/history.json

🔧 Configs:
   configs/config_v6_a10g.yaml
```

---

## 💡 Key Insight

> **Motion Features不是可选的，而是必需的。**
>
> 行为检测本质上是识别运动模式，仅靠静态坐标无法捕捉行为的动态本质。
>
> V6的成功证明：速度 + 加速度 = 42% F1提升

---

## 📞 Quick Commands

```bash
# 评估best model
modal run modal_evaluate_v6.py --checkpoint best_model.pth

# 生成Kaggle提交
python generate_submission.py --checkpoint checkpoints/v6_a10g/best_model.pth

# 下载checkpoint到本地
modal volume get mabe-data checkpoints/v6_a10g/best_model.pth ./

# 查看训练历史
modal volume get mabe-data checkpoints/v6_a10g/history.json ./
```

---

**Status**: ✅ Ready for Production
**Recommended**: Use Epoch 18 (best_model.pth) for Kaggle submission
**Created**: 2025-10-01
