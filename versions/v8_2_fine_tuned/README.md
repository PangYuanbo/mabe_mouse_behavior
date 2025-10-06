# V8.2 Fine-Tuned Post-processing

## Improvements over V8.1

### 1. Validation Smoothing Kernel = 3 ⭐
**Critical change from V8.1**

- **V8.1**: `kernel_size=5`
- **V8.2**: `kernel_size=3`

**Why**: kernel_size=5 over-smooths and loses short-duration behaviors. kernel_size=3 provides better balance between noise reduction and short-segment preservation.

**Impact**: Better detection of short behaviors (3-6 frames)

---

### 2. Lower Sniff Thresholds for Better Recall

| Behavior | V8.1 | V8.2 | Change |
|----------|------|------|--------|
| sniffgenital | 0.26 | **0.25** | ↓ 0.01 |
| sniffface | 0.28 | **0.26** | ↓ 0.02 |
| sniffbody | 0.28 | **0.25** | ↓ 0.03 |

**Why**: Sniff behaviors were under-detected in V8.1. Lowering thresholds improves recall while maintaining `frac_high_threshold=0.20` to control FP.

---

### 3. Larger sniffbody Merge Gap

- **V8.1**: `merge_gap=6`
- **V8.2**: `merge_gap=8`

**Why**: Sniffbody behaviors tend to be fragmented. Larger gap allows better segment merging.

**Impact**: Fewer fragmented sniffbody segments, more continuous intervals

---

### 4. Maintained Optimizations from V8.1

✅ **Motion Gating**:
- `escape/chase`: velocity_min = 80 px/s
- `freeze`: velocity_max = 7 px/s with threshold relaxation to 0.20

✅ **Intromit/Mount Calibration**:
- Same-pair overlap: prioritize intromit
- Shrink/split mount when overlapping

✅ **Other Settings**:
- `smoothing_kernel=3` in pipeline
- Boundary refinement `±2` frames
- Action-only segmentation + voting + merge

---

## Full Configuration

```python
CLASS_CONFIG = {
    # Sniffing behaviors (V8.2: lowered thresholds)
    'sniff': {
        'min_duration': 6,
        'prob_threshold': 0.38,
        'merge_gap': 5,
        'max_required': 0.50
    },
    'sniffgenital': {
        'min_duration': 3,
        'prob_threshold': 0.25,  # V8.2 ↓
        'merge_gap': 5,
        'frac_high_threshold': 0.20
    },
    'sniffface': {
        'min_duration': 3,
        'prob_threshold': 0.26,  # V8.2 ↓
        'merge_gap': 5,
        'frac_high_threshold': 0.20
    },
    'sniffbody': {
        'min_duration': 4,
        'prob_threshold': 0.25,  # V8.2 ↓
        'merge_gap': 8,  # V8.2 ↑
        'frac_high_threshold': 0.20
    },

    # Mating behaviors (unchanged from V8.1)
    'mount': {
        'min_duration': 4,
        'prob_threshold': 0.43,
        'merge_gap': 8
    },
    'intromit': {
        'min_duration': 3,
        'prob_threshold': 0.45,
        'merge_gap': 10
    },

    # Motion-gated behaviors (unchanged from V8.1)
    'escape': {
        'min_duration': 5,
        'prob_threshold': 0.55,
        'merge_gap': 4,
        'max_required': 0.65,
        'velocity_min': 80.0
    },
    'chase': {
        'min_duration': 5,
        'prob_threshold': 0.40,
        'merge_gap': 5,
        'velocity_min': 80.0
    },
    'freeze': {
        'min_duration': 3,
        'prob_threshold': 0.25,
        'merge_gap': 3,
        'velocity_max': 7.0
    },

    # Other behaviors (unchanged from V8.1)
    'approach': {
        'min_duration': 4,
        'prob_threshold': 0.42,
        'merge_gap': 4
    }
}
```

---

## Usage

```python
from versions.v8_2_fine_tuned.advanced_postprocessing import (
    probs_to_intervals_advanced
)

intervals = probs_to_intervals_advanced(
    action_probs=action_probs,
    agent_probs=agent_probs,
    target_probs=target_probs,
    action_names=ID_TO_ACTION,
    keypoints=keypoints_original,  # For motion gating
    smoothing_kernel=3  # V8.2: Critical parameter
)
```

---

## Expected Performance

**vs V8.1**:
- Sniffgenital/Sniffface/Sniffbody recall: **+3~8%**
- Short segment detection: **Better**
- Sniffbody continuity: **Improved**
- Overall F1: **+1~3%**

**vs V8**:
- Escape FP: **-30~50%** (motion gating)
- Freeze recall: **+10~20%** (velocity + threshold relaxation)
- Sniff recall: **+5~12%** (lower thresholds)
- Overall F1: **+3~6%**

---

## Version History

- **V8**: Baseline with basic postprocessing
- **V8.1**: Added motion gating, fine-tuned thresholds, kernel_size=5
- **V8.2**: Further lowered sniff thresholds, **kernel_size=3**, increased sniffbody merge_gap
