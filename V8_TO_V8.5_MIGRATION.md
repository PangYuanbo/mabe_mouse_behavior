# Migration Guide: V8/V8.1 → V8.5

## Executive Summary

V8.5 is a **critical compliance fix** that includes **ALL 37 behaviors** found in the training data, addressing a major flaw in V8/V8.1 which only predicted 27 behaviors.

## Why Migrate?

### Competition Requirement
> "identify over 30 different **social and non-social behaviors**"

### V8/V8.1 Issues
❌ Only 27 behaviors predicted (NUM_ACTIONS=28 including background)
❌ 11 behaviors incorrectly mapped to background
❌ 7,985 annotated intervals ignored
❌ **Non-compliant** with competition rules

### V8.5 Solution
✅ All 37 behaviors included (NUM_ACTIONS=38 including background)
✅ 0 intervals ignored
✅ **Fully compliant** with competition
✅ Better coverage of rare/non-social behaviors

## Key Differences

### Code Changes

#### 1. Imports

**V8/V8.1:**
```python
from versions.v8_fine_grained.v8_dataset import create_v8_dataloaders
from versions.v8_fine_grained.v8_model import V8BehaviorDetector, V8MultiTaskLoss
from versions.v8_fine_grained.action_mapping import NUM_ACTIONS  # 28
```

**V8.5:**
```python
from versions.v8_5_full_behaviors.v8_5_dataset import create_v85_dataloaders
from versions.v8_5_full_behaviors.v8_5_model import V85BehaviorDetector, V85MultiTaskLoss
from versions.v8_5_full_behaviors.action_mapping import NUM_ACTIONS  # 38
```

#### 2. Model Instantiation

**V8/V8.1:**
```python
model = V8BehaviorDetector(
    input_dim=112,
    num_actions=28,  # Fixed
    ...
)
```

**V8.5:**
```python
model = V85BehaviorDetector(
    input_dim=112,
    num_actions=38,  # Or use NUM_ACTIONS
    ...
)
```

#### 3. Configuration File

**V8/V8.1:** `config_v8.1_optimized.yaml`
```yaml
num_actions: 28
```

**V8.5:** `config_v8.5_full_behaviors.yaml`
```yaml
num_actions: 38
use_class_weights: true  # Essential!
class_weight_strategy: 'inverse_sqrt_freq'
```

### Behavior Mapping Changes

#### Behaviors with Changed IDs

| Behavior | V8/V8.1 ID | V8.5 ID | Reason |
|----------|------------|---------|--------|
| dominance | 16 | **15** | bite removed |
| defend | 17 | **16** | bite removed |
| flinch | 18 | **17** | bite removed |
| avoid | 19 | **18** | bite removed |
| escape | 20 | **19** | bite removed |
| freeze | 21 | **20** | bite removed |
| allogroom | 22 | **21** | bite removed |
| shepherd | 23 | **22** | bite removed |
| disengage | 24 | **23** | bite removed |
| run | 25 | **24** | bite removed |
| dominancegroom | 26 | **25** | bite removed |
| huddle | 27 | **28** | Reordered |

#### Newly Included Behaviors (Previously Mapped to 0)

| Behavior | V8/V8.1 ID | V8.5 ID | Intervals |
|----------|------------|---------|-----------|
| genitalgroom | 0 | **26** | 50 |
| selfgroom | 0 | **27** | 1,356 |
| dominancemount | 0 | **29** | 410 |
| tussle | 0 | **30** | 122 |
| rear | 0 | **31** | 4,408 |
| rest | 0 | **32** | 233 |
| dig | 0 | **33** | 1,127 |
| climb | 0 | **34** | 1,010 |
| exploreobject | 0 | **35** | 105 |
| biteobject | 0 | **36** | 33 |
| submit | 0 | **37** | 86 |

#### Removed Behaviors

| Behavior | V8/V8.1 ID | V8.5 ID | Reason |
|----------|------------|---------|--------|
| bite | 15 | **Removed** | Not in training data (0 intervals) |

## Migration Steps

### Step 1: Backup V8.1 Work

```bash
# Backup checkpoints
cp -r checkpoints/v8.1_optimized checkpoints/v8.1_optimized_backup

# Backup any custom scripts
cp train_v8.1_local.py train_v8.1_local.py.backup
```

### Step 2: Update Configuration

Create or use `configs/config_v8.5_full_behaviors.yaml`:

```yaml
num_actions: 38  # Critical change

# New: Class weights essential for 38 classes
use_class_weights: true
class_weight_strategy: 'inverse_sqrt_freq'

# Keep other V8.1 settings
use_focal_loss: true
focal_gamma: 1.5
batch_size: 256
learning_rate: 0.00005
# ...
```

### Step 3: Update Training Script

**Option A - Use provided script:**
```bash
python train_v8.5_local.py --config configs/config_v8.5_full_behaviors.yaml
```

**Option B - Modify existing script:**
Replace all V8 imports with V8.5 equivalents (see "Code Changes" above)

### Step 4: Train from Scratch

⚠️ **Important:** Cannot load V8.1 weights into V8.5
- Different output dimensions (28 → 38)
- Must retrain completely

```bash
# Start fresh training
python train_v8.5_local.py --config configs/config_v8.5_full_behaviors.yaml
```

### Step 5: Validate Results

Check that all 37 behaviors appear:
```bash
python -c "
from versions.v8_5_full_behaviors.action_mapping import ID_TO_ACTION, NUM_ACTIONS
print(f'Total classes: {NUM_ACTIONS}')
print('Behaviors:')
for i in range(1, NUM_ACTIONS):
    print(f'  {i:2d}. {ID_TO_ACTION[i]}')
"
```

## What Stays the Same?

✅ **Architecture** - Conv + LSTM structure unchanged
✅ **Input features** - 112-dim (56 coords + 28 speed + 28 accel)
✅ **Training strategy** - Focal loss, multi-task learning
✅ **Postprocessing** - Interval conversion, smoothing
✅ **Hardware requirements** - RTX 5090, batch=256
✅ **Training time** - ~Same as V8.1

## What Changes?

📝 **Output dimension** - 28 → 38 action classes
📝 **Class weights** - Now essential (more imbalance)
📝 **Evaluation** - 11 new behaviors tracked
📝 **Checkpoints** - New directory `v8.5_full_behaviors/`

## Performance Expectations

### F1 Score Changes

**Newly tracked behaviors** (11 total):
- Major impact: `rear` (4,408 intervals), `selfgroom` (1,356), `dig` (1,127), `climb` (1,010)
- Minor impact: Others (<500 intervals each)

**Expected F1:**
- **High-frequency new behaviors** (rear, selfgroom): ~0.3-0.5 F1
- **Low-frequency new behaviors** (<100 intervals): ~0.1-0.2 F1
- **Overall F1**: Modest increase (+0.02-0.05) depending on test distribution

**Existing behaviors:**
- Minimal change (same model capacity)

### Training Convergence

- **Epochs to converge**: ~Same as V8.1 (50-80 epochs)
- **Peak performance**: May appear later (more classes)
- **Validation F1**: May fluctuate more (11 new classes)

## Troubleshooting

### Error: "Model checkpoint incompatible"
**Cause:** Trying to load V8.1 weights into V8.5
**Solution:** Train from scratch, no weight transfer possible

### Error: "NUM_ACTIONS mismatch (expected 28, got 38)"
**Cause:** Mixed V8 and V8.5 imports
**Solution:** Check all imports use `v8_5_full_behaviors`

### Warning: "Very low F1 on rare behaviors"
**Expected:** Behaviors with <50 intervals are hard to learn
**Solutions:**
- Increase `focal_gamma` to 2.0
- Use `class_weight_strategy: 'inverse_freq'`
- Train longer (100 epochs)

### Issue: "Class weights not applied"
**Cause:** `use_class_weights: false` in config
**Solution:** Set `use_class_weights: true`

## Validation Checklist

After migration, verify:

- [ ] NUM_ACTIONS = 38 (check with `print(NUM_ACTIONS)`)
- [ ] Model output shape = [B, T, 38]
- [ ] Config uses `config_v8.5_full_behaviors.yaml`
- [ ] Class weights computed (shows 38 classes during training)
- [ ] All 37 behaviors in evaluation output
- [ ] No "ignored intervals" warnings
- [ ] Checkpoints saved to `checkpoints/v8.5_full_behaviors/`

## Comparison Table

| Feature | V8.1 | V8.5 | Impact |
|---------|------|------|--------|
| **Behaviors** | 27 | 37 | ⚠️ Critical |
| **NUM_ACTIONS** | 28 | 38 | ⚠️ Critical |
| **Ignored intervals** | 7,985 | 0 | ⚠️ Critical |
| **Competition compliant** | ❌ | ✅ | ⚠️ Critical |
| **Non-social behaviors** | ❌ | ✅ | ⚠️ Critical |
| **Architecture** | Same | Same | ✅ No change |
| **Input dim** | 112 | 112 | ✅ No change |
| **Training time** | ~X hrs | ~X hrs | ✅ Minimal |
| **Hardware** | 5090 | 5090 | ✅ No change |
| **Class weights** | Optional | **Essential** | ⚠️ Required |

## FAQ

### Q: Can I use V8.1 weights for fine-tuning?
**A:** No. Output dimensions are different (28 vs 38). Must train from scratch.

### Q: Will V8.5 perform better than V8.1?
**A:** On newly included behaviors: **Yes**. Overall: **Depends on test set**. But V8.5 is **required for competition compliance**.

### Q: How long does migration take?
**A:** Code changes: 5 min. Retraining: Same as V8.1 (~X hours).

### Q: Can I keep V8.1 running in parallel?
**A:** Yes! V8.5 uses separate directories. You can compare results.

### Q: Should I migrate immediately?
**A:** **Yes**, if you care about competition compliance and covering all behaviors.

## Resources

- **V8.5 README**: `versions/v8_5_full_behaviors/README.md`
- **Quick Start**: `V8.5_QUICK_START.md`
- **Competition Rules**: `COMPETITION_RULES.md`
- **Label Analysis**: `analyze_label_distribution_detailed.py`
- **Training Script**: `train_v8.5_local.py`

## Getting Help

**Check first:**
1. V8.5 README
2. This migration guide
3. Quick start guide

**Common issues:**
- Import errors → Check imports use `v8_5_full_behaviors`
- Weight loading errors → Cannot load V8.1 weights, train from scratch
- Low F1 on new behaviors → Expected for rare behaviors (<50 intervals)

---

**Ready to migrate? Run:**
```bash
python train_v8.5_local.py --config configs/config_v8.5_full_behaviors.yaml
```
