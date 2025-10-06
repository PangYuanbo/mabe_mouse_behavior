"""
V8.5 Full Behavior Coverage
ALL 37 behaviors included (not just 27 like V8/V8.1)

Key improvements:
1. NUM_ACTIONS = 38 (0=background + 37 behaviors)
2. Includes non-social behaviors: rear, selfgroom, dig, climb, etc.
3. Removed 'bite' (not in training data)
4. Competition-compliant: "identify over 30 different social and non-social behaviors"
5. Uses class weights and focal loss for severe imbalance
"""

from .action_mapping import (
    ACTION_TO_ID,
    ID_TO_ACTION,
    NUM_ACTIONS,
    ACTION_CATEGORIES,
    FREQUENCY_GROUPS,
    get_action_id,
    get_action_name,
    get_category,
    get_frequency_group
)

from .v8_5_model import (
    V85BehaviorDetector,
    V85MultiTaskLoss,
    FocalLoss
)

from .v8_5_dataset import (
    V85Dataset,
    create_v85_dataloaders
)

__all__ = [
    # Mapping
    'ACTION_TO_ID',
    'ID_TO_ACTION',
    'NUM_ACTIONS',
    'ACTION_CATEGORIES',
    'FREQUENCY_GROUPS',
    'get_action_id',
    'get_action_name',
    'get_category',
    'get_frequency_group',
    # Model
    'V85BehaviorDetector',
    'V85MultiTaskLoss',
    'FocalLoss',
    # Dataset
    'V85Dataset',
    'create_v85_dataloaders',
]
