# V8.5 - Full Behavior Coverage (ALL 37 Behaviors)

## 🎯 Mission Critical Update

**V8.5 is a competition-compliance fix for V8/V8.1**

### The Problem with V8/V8.1

V8 and V8.1 had a **critical flaw**: they only predicted **27 behaviors**, mapping 11 behaviors to background:
- `rear` (4,408 intervals) ❌
- `selfgroom` (1,356 intervals) ❌
- `dig` (1,127 intervals) ❌
- `climb` (1,010 intervals) ❌
- `rest`, `exploreobject`, `biteobject`, `genitalgroom`, `dominancemount`, `submit`, `tussle` ❌

**Total ignored: 7,985 annotated intervals**

But the competition explicitly states:
> "identify over 30 different **social and non-social behaviors**"

### The V8.5 Solution

✅ **Includes ALL 37 behaviors** found in training data
✅ **Non-social behaviors** now have unique class IDs
✅ **Removed `bite`** (ID 15) - not present in data
✅ **NUM_ACTIONS = 38** (0=background + 37 behaviors)
✅ **Fully competition-compliant**

## 📊 Behavior Mapping

### V8/V8.1 vs V8.5 Comparison

| Behavior | V8/V8.1 | V8.5 | Intervals | Category |
|----------|---------|------|-----------|----------|
| sniff | ID 1 | ID 1 | 37,837 | Social |
| rear | **ID 0 (❌)** | **ID 31 (✅)** | 4,408 | Individual |
| selfgroom | **ID 0 (❌)** | **ID 27 (✅)** | 1,356 | Grooming |
| dig | **ID 0 (❌)** | **ID 33 (✅)** | 1,127 | Individual |
| climb | **ID 0 (❌)** | **ID 34 (✅)** | 1,010 | Individual |
| bite | **ID 15** | **Removed** | 0 | N/A |

### Complete V8.5 Behavior List (37 behaviors)

**Social Investigation (7):**
1. sniff
2. sniffgenital
3. sniffface
4. sniffbody
5. reciprocalsniff
6. approach
7. follow

**Mating (4):**
8. mount
9. intromit
10. attemptmount
11. ejaculate

**Aggressive (6):**
12. attack
13. chase
14. chaseattack
15. dominance *(was 16)*
16. defend *(was 17)*
17. flinch *(was 18)*

**Social Other (7):**
18. avoid *(was 19)*
19. escape *(was 20)*
20. freeze *(was 21)*
21. allogroom *(was 22)*
22. shepherd *(was 23)*
23. disengage *(was 24)*
24. run *(was 25)*

**Grooming (3):** *(NEW in V8.5!)*
25. dominancegroom *(was 26)*
26. genitalgroom *(was 0)*
27. selfgroom *(was 0)*

**Group/Contact (3):**
28. huddle *(was 27)*
29. dominancemount *(was 0)*
30. tussle *(was 0)*

**Individual (7):** *(NEW in V8.5!)*
31. rear *(was 0)*
32. rest *(was 0)*
33. dig *(was 0)*
34. climb *(was 0)*
35. exploreobject *(was 0)*
36. biteobject *(was 0)*
37. submit *(was 0)*

## 🚀 Quick Start

### Training

```bash
python train_v8.5_local.py --config configs/config_v8.5_full_behaviors.yaml
```

### Key Configuration Changes

```yaml
num_actions: 38  # Was 28 in V8/V8.1

# Class weights critical for 38 classes!
use_class_weights: true
class_weight_strategy: 'inverse_sqrt_freq'

# Focal loss for severe imbalance
use_focal_loss: true
focal_gamma: 1.5
```

## 🎓 Technical Details

### Architecture

**Same as V8**, just different output dimension:
- Conv backbone: [128, 256, 512]
- BiLSTM: 256 hidden, 2 layers
- Action head: **38 classes** (was 28)
- Agent/Target heads: 4 mice (unchanged)

### Loss Function

**Multi-task loss with class weights:**
```python
V85MultiTaskLoss(
    action_weight=1.0,
    agent_weight=0.3,
    target_weight=0.3,
    use_focal=True,
    focal_gamma=1.5,
    class_weights=[38]  # Computed from frequency
)
```

### Class Imbalance Strategy

With 38 classes, imbalance is **severe**:
- Background: ~72% of frames
- `sniff`: 11.33%
- `ejaculate`: 0.008% (only 3 intervals!)

**V8.5 handles this with:**
1. **Focal Loss** (γ=1.5) - focuses on hard examples
2. **Class Weights** - inverse sqrt frequency
3. **Adaptive min durations** - per-behavior thresholds

### Data Statistics

| Metric | V8/V8.1 | V8.5 | Change |
|--------|---------|------|--------|
| Behaviors | 27 | **37** | +10 (+37%) |
| Ignored intervals | 7,985 | **0** | -100% |
| Model classes | 28 | **38** | +10 |
| Params | ~8.5M | ~8.5M | +0.001% (output layer only) |

## 📈 Expected Improvements

### Coverage
- **+37% behavior coverage** (27→37 behaviors)
- **0 ignored intervals** (was 7,985)

### F1 Score
- Individual behaviors: **+significant** (rear, selfgroom, dig, climb now evaluated)
- Overall: **+modest** (depends on test set distribution)

### Competition Compliance
- **Fully compliant** with "30+ behaviors" requirement
- **No risk** of missing behaviors in test set

## 🔄 Migration from V8.1

### For Training

```python
# OLD (V8.1)
from versions.v8_fine_grained.v8_dataset import create_v8_dataloaders
from versions.v8_fine_grained.v8_model import V8BehaviorDetector
from versions.v8_fine_grained.action_mapping import NUM_ACTIONS  # 28

# NEW (V8.5)
from versions.v8_5_full_behaviors.v8_5_dataset import create_v85_dataloaders
from versions.v8_5_full_behaviors.v8_5_model import V85BehaviorDetector
from versions.v8_5_full_behaviors.action_mapping import NUM_ACTIONS  # 38
```

### For Inference

**Cannot directly use V8.1 weights for V8.5!**
- Different output dimensions (28 vs 38)
- Must retrain from scratch

## ⚠️ Important Notes

### 1. Retraining Required
V8.5 requires **complete retraining** - cannot load V8.1 weights

### 2. Class Weights Essential
With 38 classes and severe imbalance, class weights are **critical**

### 3. Longer Training
More classes may require more epochs to converge

### 4. Evaluation Changes
Per-class metrics will show **11 new behaviors** (previously ignored)

## 📁 File Structure

```
versions/v8_5_full_behaviors/
├── action_mapping.py      # All 37 behaviors
├── v8_5_model.py          # 38-class output
├── v8_5_dataset.py        # Uses new mapping
├── submission_utils.py    # Interval conversion
├── __init__.py
└── README.md
```

## 🎯 Competition Alignment

### Competition Requirement
> "identify over 30 different social and non-social behaviors"

### V8.1 Status
❌ Only 27 behaviors
❌ Non-social behaviors mapped to background
❌ **Not fully compliant**

### V8.5 Status
✅ **37 behaviors** (exceeds 30)
✅ **Social AND non-social** included
✅ **Fully compliant**

## 🔬 Testing

```bash
# Test action mapping
python -c "from versions.v8_5_full_behaviors.action_mapping import print_mapping_summary; print_mapping_summary()"

# Should output:
# Total classes: 38 (0=background, 1-37=behaviors)
# Behaviors by category: ...
```

## 📚 References

- **Competition Rules**: `COMPETITION_RULES.md`
- **Label Analysis**: `analyze_label_distribution_detailed.py`
- **V8.1**: `versions/v8_1_optimized_postproc/` (preserved)

## 🚦 Status

- ✅ Code complete
- ✅ Config ready
- ⏳ Training pending
- ⏳ Validation pending

---

**V8.5: Because every behavior matters. 🐭**
