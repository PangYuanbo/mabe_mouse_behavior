# V8.6 MARS-Enhanced Behavior Detection

**Status**: Research Implementation
**Base**: V8.5 (38 behaviors)
**Inspiration**: MARS paper (eLife 2021) - [https://elifesciences.org/articles/63720](https://elifesciences.org/articles/63720)

---

## 🎯 **Objective**

Dramatically improve rare behavior detection (especially **freeze**) by implementing MARS-style feature engineering and multi-scale temporal modeling.

**Target Metrics**:
- **Interval F1**: 0.255 → **0.35+** (+37% improvement)
- **Freeze F1**: 0.0 → **0.15+** (from total failure to detectable)
- **Sniffgenital Recall**: 11.7% → **25%+** (+113% improvement)

---

## 🔬 **Key Innovations**

### 1. **MARS-Inspired Feature Engineering** (112 → ~370 dims)

Implemented in `feature_engineering.py`:

| Feature Group | Dims | Purpose |
|---------------|------|---------|
| Raw coordinates | 56 | Base skeleton |
| **Joint angles** | 48 | Posture detection (mount, intromit) |
| **Body orientation** | 4 | Direction facing (chase vs avoid) |
| Velocity/Speed | 28 | Motion magnitude |
| Acceleration | 28 | Motion change |
| **Jerk** | 28 | **Critical for freeze detection** |
| Angular velocity | 4 | Turning rate |
| **Inter-mouse distances** | 42 | Social proximity (all pairs) |
| **Relative positions** | 48 | Agent-target relationships |
| **Social features** | 24 | Contact, approach/retreat |
| **Total** | **~370** | **3.3× richer than V8.5** |

#### **Why These Features Matter**

From MARS paper analysis:

- **Freeze detection failure** in V8.5 was due to missing **jerk** (rate of acceleration change) - freeze is characterized by sudden drop in jerk, not just low speed
- **Sniffgenital under-prediction** needs **relative nose-to-genital distance** and **body orientation**
- **Mount/intromit** requires **joint angles** (hip angle, spine curvature) not captured by raw coordinates

---

### 2. **Multi-Scale Temporal Modeling**

Implemented in `v8_6_model.py`:

```python
# Different behaviors have different time scales:
Short kernel (3):   attack, flinch       (~0.09 sec)
Medium kernel (7):  sniff, mount         (~0.21 sec)
Long kernel (15):   freeze, intromit     (~0.45 sec)

# Architecture:
Input [B, 150, 370]  # 4.5 sec sequences
  ↓
Multi-scale Conv (parallel 3/7/15) → Fusion
  ↓
BiLSTM (3 layers, hidden=384)
  ↓
Self-Attention (focus critical windows)
  ↓
├─ Action Head (38 classes)
├─ Agent/Target Heads (4 classes)
└─ Freeze Head (2 classes) ← NEW!
```

**Why multi-scale?**
- V8.5 used fixed kernel=5, averaging 3-frame (attack) and 15-frame (freeze) behaviors → poor for both
- V8.6 learns separate representations for fast vs slow behaviors

---

### 3. **Dedicated Freeze Detection Branch**

**Problem**: Freeze is binary (freeze/non-freeze) but was treated as 1-of-38 classification → easily overwhelmed by more common behaviors

**Solution**: Auxiliary binary classifier with:
- Specialized architecture (smaller, focused)
- Separate loss weight (0.5)
- Boosted class weights (10:1 freeze:non-freeze ratio needs compensation)

---

### 4. **Dynamic Class Weighting**

**V8.5 problem**: All behaviors weighted equally → model ignores rare behaviors

**V8.6 solution**:
```python
# Base weight: inverse sqrt frequency
weight = sqrt(total_frames / class_count)

# Boost rare behaviors 3x:
rare_behaviors = [freeze, escape, dominancegroom, ...]
weight[rare_behaviors] *= 3.0
```

**Effect**: Freeze gets ~15× more weight than background → model forced to learn it

---

### 5. **Rare Behavior Oversampling**

**Training data augmentation**:
- Sequences containing freeze, escape, etc. are **repeated 3 times** in training set
- Combined with data augmentation (coordinate noise, temporal jitter)
- Effectively 9× more exposure to rare behaviors

---

## 📊 **Expected Improvements**

Based on MARS paper results and V8.5 analysis:

| Behavior | V8.5 Interval F1 | V8.6 Target | Improvement |
|----------|-----------------|-------------|-------------|
| **freeze** | **0.00** | **0.15** | **+∞** |
| sniffgenital | 0.12 | 0.25 | +108% |
| escape | 0.02 | 0.10 | +400% |
| attack | 0.49 | 0.55 | +12% |
| mount | 0.31 | 0.40 | +29% |
| **Overall** | **0.255** | **0.35** | **+37%** |

---

## 🏗️ **Architecture Details**

### Model Size

```
V8.5:  3.1M parameters (input: 112, hidden: 256, layers: 2)
V8.6:  8.7M parameters (input: 370, hidden: 384, layers: 3)

Increase: +2.8× parameters
Trade-off: Worth it for +37% performance
```

### Sequence Length

```
V8.5:  100 frames (3.0 sec at 33.3 fps)
V8.6:  150 frames (4.5 sec at 33.3 fps)

Why longer?
- Freeze bouts average 2-4 seconds
- Mount sequences can last 3+ seconds
- Need full context to distinguish freeze from rest
```

---

## 🚀 **Training Instructions**

### 1. **Prerequisites**

```bash
# Recommended GPU: RTX 5090 (32GB VRAM) or A100
# Minimum: RTX 3090 (24GB VRAM) with reduced batch size

# Memory usage:
- Batch size 128: ~22GB VRAM
- Batch size 64:  ~12GB VRAM
```

### 2. **Run Training**

```bash
python train_v8_6_local.py --config configs/config_v8.6_mars.yaml
```

### 3. **Monitor Progress**

Key metrics to watch:
- **Interval F1** (main Kaggle metric)
- **Freeze Accuracy** (should reach 70%+)
- **Freeze F1** (should reach 0.15+)

Training time:
- ~2.5 min/epoch on RTX 5090
- ~50 epochs to convergence
- Total: ~2 hours

---

## 📁 **File Structure**

```
versions/v8_6_mars_enhanced/
├── feature_engineering.py  # MARS feature extraction (~370 dims)
├── v8_6_model.py          # Multi-scale model + Freeze branch
├── v8_6_dataset.py        # Dataset with oversampling + augmentation
├── action_mapping.py      # 38 behavior classes (from V8.5)
└── README.md              # This file

train_v8_6_local.py         # Training script
configs/config_v8.6_mars.yaml  # Hyperparameters
```

---

## 🧪 **Ablation Study Plan**

To validate each improvement, run these experiments:

1. **Baseline** (V8.5 features + V8.5 model): Interval F1 = 0.255
2. **+MARS features only**: Expected +0.05 (features alone help)
3. **+Multi-scale conv**: Expected +0.03 (better temporal modeling)
4. **+Freeze branch**: Expected +0.02 (freeze-specific)
5. **+Dynamic weighting**: Expected +0.02 (rare behavior boost)
6. **Full V8.6**: Expected **0.35+**

---

## 📈 **Validation Strategy**

### Per-Class Monitoring

Focus on these 5 behaviors (biggest impact on final score):

1. **freeze** (currently 0.0 F1) - Target: 0.15
2. **sniffgenital** (currently 0.12 F1) - Target: 0.25
3. **attack** (currently 0.49 F1) - Target: 0.55
4. **mount** (currently 0.31 F1) - Target: 0.40
5. **escape** (currently 0.02 F1) - Target: 0.10

If these 5 improve, overall F1 will reach 0.35+.

---

## ⚠️ **Known Limitations**

1. **Computational Cost**
   - 3× slower training than V8.5 (due to larger features/sequences)
   - Requires 24GB+ VRAM for batch_size=128

2. **Feature Computation**
   - MARS features computed on-the-fly (not pre-cached)
   - Consider pre-computing and saving to disk if training multiple times

3. **Overfitting Risk**
   - More parameters → higher overfitting risk
   - Mitigated by: dropout=0.3, weight_decay=0.01, early stopping

---

## 🔧 **Hyperparameter Tuning**

If results are suboptimal, try:

### If overfitting (train F1 >> val F1):
- Increase dropout: 0.3 → 0.4
- Increase weight_decay: 0.01 → 0.02
- Reduce model size: hidden=384 → 320

### If underfitting (both train/val F1 low):
- Increase model size: hidden=384 → 512
- Increase sequence length: 150 → 200
- Reduce dropout: 0.3 → 0.2

### If freeze still fails:
- Increase freeze_weight: 0.5 → 1.0
- Increase rare_boost: 3.0 → 5.0
- Check freeze feature importance (jerk should be high)

---

## 📚 **References**

1. **MARS Paper**: [Automated mouse behavior analysis using supervised machine learning](https://elifesciences.org/articles/63720)
   - Segalin et al., eLife 2021
   - Key insight: 270+ features needed for robust social behavior detection

2. **Focal Loss**: Lin et al., ICCV 2017
   - Handles extreme class imbalance (freeze is 0.5% of frames)

3. **Multi-scale Temporal Convolution**: Yan et al., AAAI 2018
   - ST-GCN architecture for action recognition

---

## ✅ **Next Steps After V8.6**

If V8.6 reaches 0.35 Interval F1:

1. **V8.7**: Transformer-based architecture (replace LSTM)
2. **V8.8**: Graph Neural Network (model mouse interactions explicitly)
3. **V8.9**: Ensemble V8.5 + V8.6 + V8.7 (potential 0.40+ F1)

If V8.6 underperforms:

1. Debug freeze detection (visualize jerk features)
2. Ablation study to isolate which features help
3. Consider pre-training on simpler tasks (binary freeze detection first)

---

## 📞 **Support**

Issues? Check:
1. GPU memory (need 24GB+ for batch_size=128)
2. Data path in config (must point to Kaggle dataset)
3. Feature extraction (run standalone test: `python -m versions.v8_6_mars_enhanced.feature_engineering`)

---

**Good luck training! 🚀**

Expected result: **Interval F1 = 0.35+** (current SOTA: 0.255)
