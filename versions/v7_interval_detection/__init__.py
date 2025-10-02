"""
V7: Temporal Action Detection for Mouse Behavior
Direct interval prediction instead of frame-level classification
"""

from .interval_dataset import IntervalDetectionDataset, collate_interval_fn
from .interval_model import TemporalActionDetector
from .interval_loss import IntervalDetectionLoss
from .interval_metrics import IntervalMetrics, evaluate_intervals

__all__ = [
    'IntervalDetectionDataset',
    'collate_interval_fn',
    'TemporalActionDetector',
    'IntervalDetectionLoss',
    'IntervalMetrics',
    'evaluate_intervals',
]
