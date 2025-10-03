"""
V8 Action Mapping - 31 Fine-grained Behavior Classes
Maps MABe behavior names to class IDs for Kaggle competition
"""

# Action name to ID mapping (31 classes including background)
ACTION_TO_ID = {
    # Background (0)
    'background': 0,

    # Social investigation (1-7)
    'sniff': 1,
    'sniffgenital': 2,
    'sniffface': 3,
    'sniffbody': 4,
    'reciprocalsniff': 5,
    'approach': 6,
    'follow': 7,

    # Mating behaviors (8-11)
    'mount': 8,
    'intromit': 9,
    'attemptmount': 10,
    'ejaculate': 11,

    # Aggressive behaviors (12-18)
    'attack': 12,
    'chase': 13,
    'chaseattack': 14,
    'bite': 15,
    'dominance': 16,
    'defend': 17,
    'flinch': 18,

    # Other social behaviors (19-30)
    'avoid': 19,
    'escape': 20,
    'freeze': 21,
    'allogroom': 22,
    'shepherd': 23,
    'disengage': 24,
    'run': 25,
    'dominancegroom': 26,
    'huddle': 27,

    # Non-social behaviors (mapped to background)
    'rear': 0,
    'selfgroom': 0,
    'rest': 0,
    'dig': 0,
    'climb': 0,
    'exploreobject': 0,
    'biteobject': 0,
    'other': 0,
}

# Reverse mapping: ID to action name
ID_TO_ACTION = {
    0: 'background',
    1: 'sniff',
    2: 'sniffgenital',
    3: 'sniffface',
    4: 'sniffbody',
    5: 'reciprocalsniff',
    6: 'approach',
    7: 'follow',
    8: 'mount',
    9: 'intromit',
    10: 'attemptmount',
    11: 'ejaculate',
    12: 'attack',
    13: 'chase',
    14: 'chaseattack',
    15: 'bite',
    16: 'dominance',
    17: 'defend',
    18: 'flinch',
    19: 'avoid',
    20: 'escape',
    21: 'freeze',
    22: 'allogroom',
    23: 'shepherd',
    24: 'disengage',
    25: 'run',
    26: 'dominancegroom',
    27: 'huddle',
}

NUM_ACTIONS = 28  # 0-27 (background + 27 behaviors)

# Action categories for analysis
ACTION_CATEGORIES = {
    'social': [1, 2, 3, 4, 5, 6, 7],
    'mating': [8, 9, 10, 11],
    'aggressive': [12, 13, 14, 15, 16, 17, 18],
    'other': [19, 20, 21, 22, 23, 24, 25, 26, 27],
}

def get_action_id(action_name: str) -> int:
    """Get action ID from name"""
    return ACTION_TO_ID.get(action_name, 0)

def get_action_name(action_id: int) -> str:
    """Get action name from ID"""
    return ID_TO_ACTION.get(action_id, 'background')

def get_category(action_id: int) -> str:
    """Get category name for an action ID"""
    if action_id == 0:
        return 'background'
    for cat, ids in ACTION_CATEGORIES.items():
        if action_id in ids:
            return cat
    return 'unknown'
