"""
V8 Fine-grained Behavior Detection Package
Multi-task learning for Kaggle MABe competition
"""

from .action_mapping import (
    ACTION_TO_ID,
    ID_TO_ACTION,
    NUM_ACTIONS,
    ACTION_CATEGORIES,
    get_action_id,
    get_action_name,
    get_category
)

from .v8_model import (
    V8BehaviorDetector,
    FocalLoss,
    V8MultiTaskLoss
)

from .submission_utils import (
    predictions_to_intervals,
    create_submission,
    merge_overlapping_intervals
)

__version__ = '8.0.0'
__all__ = [
    # Action mapping
    'ACTION_TO_ID',
    'ID_TO_ACTION',
    'NUM_ACTIONS',
    'ACTION_CATEGORIES',
    'get_action_id',
    'get_action_name',
    'get_category',

    # Models
    'V8BehaviorDetector',
    'FocalLoss',
    'V8MultiTaskLoss',

    # Submission utils
    'predictions_to_intervals',
    'create_submission',
    'merge_overlapping_intervals',
]
