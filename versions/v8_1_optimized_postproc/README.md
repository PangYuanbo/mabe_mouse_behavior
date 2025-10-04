# V8.1 Optimized Post-processing

## Improvements over V8

### 1. Motion Gating (Speed Threshold Filtering)
**Critical for reducing FP in high/low-speed behaviors**

- **High-speed behaviors (escape/chase)**:
  - `velocity_min = 80 px/s` (agent must move fast enough)
  - Discards false positives where mouse is stationary/slow
  - **Escape**: Major FP reduction expected

- **Low-speed behaviors (freeze)**:
  - `velocity_max = 7 px/s` (agent must be nearly still)
  - **Threshold relaxation**: If velocity ≤ 7 px/s, accept `prob ≥ 0.20` (instead of 0.25)
  - Improves recall for genuine freeze events

### 2. Fine-tuned Class Thresholds

#### Sniffing
- `sniff`: `max_required = 0.50` (↑ for FP control)
- `sniffgenital`: `prob_threshold = 0.26` (↓ for recall)
- `sniffbody`: `prob_threshold = 0.28`, `merge_gap = 6` (↑ for easier merging)

#### Mating
- `intromit`: `prob_threshold = 0.45` (↑ for precision)
- `mount`: `prob_threshold = 0.43` (↑ slightly)
- Calibration logic: prioritize intromit when overlapping with mount

#### Social
- `approach`: `prob_threshold = 0.42` (↑ to control noise)

#### Motion-critical
- `escape`: `prob_threshold = 0.55`, `max_required = 0.65`, **`velocity_min = 80`**
- `chase`: **`velocity_min = 80`**
- `freeze`: **`velocity_max = 7`** with threshold relaxation

### 3. Decoding Settings (Optimized)
- **Smoothing kernel**: `3` (balanced, short-segment friendly)
- **Boundary snap window**: `±2` frames (avoids overshooting)
- **Pipeline**: action-only segmentation → voting → merge → boundary refine

## Usage

Replace import in your training/inference script:
```python
# Old
from versions.v8_fine_grained.advanced_postprocessing import probs_to_intervals_advanced

# New
from versions.v8.1_optimized_postproc.advanced_postprocessing import probs_to_intervals_advanced
```

The function signature is identical; existing code will work without changes.

## Expected Improvements
- **Escape**: Significantly fewer false positives (velocity gating)
- **Freeze**: Better recall for genuine freeze (relaxed threshold when slow)
- **Intromit/Mount**: Higher precision (stricter thresholds + calibration)
- **Sniffbody**: Better segment merging (larger gap tolerance)
