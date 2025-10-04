"""
V9 Interval Assembler - Method 1: Learnable Boundary Detector

Converts V8 frame-level predictions to high-quality intervals using
learnable boundary detection heads.
"""

from .pair_mapping import get_pair_id, get_agent_target_from_pair_id, get_channel_index
from .boundary_labels import generate_soft_boundary_labels, generate_hard_boundary_labels
from .assembler_model import IntervalAssembler
from .assembler_loss import AssemblerLoss, FocalLoss, SoftIoULoss
from .decoder import decode_intervals, temporal_nms
from .v9_dataset import V9Dataset, create_v9_dataloaders

__all__ = [
    'get_pair_id',
    'get_agent_target_from_pair_id',
    'get_channel_index',
    'generate_soft_boundary_labels',
    'generate_hard_boundary_labels',
    'IntervalAssembler',
    'AssemblerLoss',
    'FocalLoss',
    'SoftIoULoss',
    'decode_intervals',
    'temporal_nms',
    'V9Dataset',
    'create_v9_dataloaders',
]
