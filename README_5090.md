# MABe Mouse Behavior - RTX 5090 Training Guide

## ✨ Quick Start (5090)

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pandas pyarrow scikit-learn scipy pyyaml tqdm
```

### 2. Quick Test (~2 min)

```bash
python train_local_5090.py --max-sequences 10
```

### 3. Full Training (~6 hours)

```bash
python train_local_5090.py
```

---

## 📊 What You Get

| Feature | Value |
|---------|-------|
| **Input Dimension** | 288 (coords + speed + accel) |
| **Batch Size** | 96 (optimized for 32GB) |
| **Expected F1** | **0.40~0.50** (vs current 0.31) |
| **Training Time** | **6 hours** (vs 12h on Modal) |
| **Cost** | **$0** (vs $13 on Modal) |

---

## 📂 Key Files

```
configs/config_5090.yaml          # 5090 optimized config
train_local_5090.py               # Local training script
src/data/kaggle_dataset.py        # With motion features
OPTIMIZATION_V2_SUMMARY.md        # Full documentation
```

---

## 🎯 Expected Results

| Behavior | Current | Expected |
|----------|---------|----------|
| Aggressive | 0.22 | **0.35~0.45** ⬆️ |
| Social | 0.36 | **0.40~0.48** ⬆️ |
| Mating | 0.40 | **0.42~0.48** ⬆️ |
| **F1 Macro** | 0.31 | **0.40~0.50** 🎯 |

**Goal**: Beat leaderboard #1 (0.40)

---

## 📖 Full Documentation

See `OPTIMIZATION_V2_SUMMARY.md` for complete details.
