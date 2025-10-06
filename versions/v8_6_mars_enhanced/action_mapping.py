"""
V8.5 Action Mapping - ALL 37 Behaviors (Full Competition-Compliant)

Key changes from V8/V8.1:
1. Includes ALL 37 behaviors found in training data (not just 27)
2. Non-social behaviors now have their own class IDs (not mapped to background)
3. Removed 'bite' (ID 15) - not present in training data
4. NUM_ACTIONS = 38 (0=background + 37 behaviors)

This mapping fully complies with competition requirement:
"identify over 30 different social and non-social behaviors"
"""

# Action name to ID mapping (38 classes: 0=background, 1-37=behaviors)
ACTION_TO_ID = {
    # Background (0) - frames with NO behavior occurring
    'background': 0,

    # ===== SOCIAL INVESTIGATION (1-7) =====
    'sniff': 1,                    # 37,837 intervals - most common
    'sniffgenital': 2,             # 7,862 intervals
    'sniffface': 3,                # 2,811 intervals
    'sniffbody': 4,                # 3,518 intervals
    'reciprocalsniff': 5,          # 1,492 intervals
    'approach': 6,                 # 3,270 intervals
    'follow': 7,                   # 233 intervals

    # ===== MATING BEHAVIORS (8-11) =====
    'mount': 8,                    # 2,747 intervals
    'intromit': 9,                 # 691 intervals
    'attemptmount': 10,            # 223 intervals
    'ejaculate': 11,               # 3 intervals - extremely rare!

    # ===== AGGRESSIVE BEHAVIORS (12-17) =====
    # NOTE: 'bite' (previously ID 15) removed - not in training data
    'attack': 12,                  # 7,462 intervals
    'chase': 13,                   # 826 intervals
    'chaseattack': 14,             # 124 intervals
    'dominance': 15,               # 329 intervals (was 16)
    'defend': 16,                  # 1,409 intervals (was 17)
    'flinch': 17,                  # 184 intervals (was 18)

    # ===== OTHER SOCIAL BEHAVIORS (18-24) =====
    'avoid': 18,                   # 530 intervals (was 19)
    'escape': 19,                  # 2,071 intervals (was 20)
    'freeze': 20,                  # 105 intervals (was 21)
    'allogroom': 21,               # 45 intervals (was 22)
    'shepherd': 22,                # 201 intervals (was 23)
    'disengage': 23,               # 279 intervals (was 24)
    'run': 24,                     # 76 intervals (was 25)

    # ===== GROOMING BEHAVIORS (25-27) =====
    # V8.5: Now included (previously mapped to background!)
    'dominancegroom': 25,          # 53 intervals (was 26)
    'genitalgroom': 26,            # 50 intervals (was mapped to 0)
    'selfgroom': 27,               # 1,356 intervals (was mapped to 0)

    # ===== GROUP/CONTACT BEHAVIORS (28-30) =====
    'huddle': 28,                  # 299 intervals (was 27)
    'dominancemount': 29,          # 410 intervals (was mapped to 0)
    'tussle': 30,                  # 122 intervals (was mapped to 0)

    # ===== NON-SOCIAL INDIVIDUAL BEHAVIORS (31-37) =====
    # V8.5: Now included (previously mapped to background!)
    'rear': 31,                    # 4,408 intervals (was mapped to 0)
    'rest': 32,                    # 233 intervals (was mapped to 0)
    'dig': 33,                     # 1,127 intervals (was mapped to 0)
    'climb': 34,                   # 1,010 intervals (was mapped to 0)
    'exploreobject': 35,           # 105 intervals (was mapped to 0)
    'biteobject': 36,              # 33 intervals (was mapped to 0)
    'submit': 37,                  # 86 intervals (was mapped to 0)

    # Fallback
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
    15: 'dominance',
    16: 'defend',
    17: 'flinch',
    18: 'avoid',
    19: 'escape',
    20: 'freeze',
    21: 'allogroom',
    22: 'shepherd',
    23: 'disengage',
    24: 'run',
    25: 'dominancegroom',
    26: 'genitalgroom',
    27: 'selfgroom',
    28: 'huddle',
    29: 'dominancemount',
    30: 'tussle',
    31: 'rear',
    32: 'rest',
    33: 'dig',
    34: 'climb',
    35: 'exploreobject',
    36: 'biteobject',
    37: 'submit',
}

NUM_ACTIONS = 38  # 0-37 (background + 37 behaviors)

# Action categories for analysis
ACTION_CATEGORIES = {
    'social_investigation': [1, 2, 3, 4, 5, 6, 7],
    'mating': [8, 9, 10, 11],
    'aggressive': [12, 13, 14, 15, 16, 17],
    'social_other': [18, 19, 20, 21, 22, 23, 24],
    'grooming': [25, 26, 27],
    'group_contact': [28, 29, 30],
    'individual': [31, 32, 33, 34, 35, 36, 37],
}

# Frequency-based categorization (for training strategies)
FREQUENCY_GROUPS = {
    'very_high': [1, 2, 12, 31],              # >3000 intervals
    'high': [3, 4, 6, 8, 19, 27, 33, 34],     # 1000-3000 intervals
    'medium': [5, 7, 13, 16, 29],              # 200-1000 intervals
    'low': [9, 10, 14, 15, 17, 18, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 35, 37],  # 50-200 intervals
    'very_low': [11, 36],                      # <50 intervals (challenging!)
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

def get_frequency_group(action_id: int) -> str:
    """Get frequency group for an action ID"""
    for group, ids in FREQUENCY_GROUPS.items():
        if action_id in ids:
            return group
    return 'unknown'

def get_v8_to_v85_mapping():
    """
    Get mapping from V8/V8.1 action IDs to V8.5 action IDs

    Returns:
        dict: {v8_id: v8.5_id}
    """
    # V8/V8.1 had 28 classes (0-27), some behaviors mapped to 0
    # V8.5 has 38 classes (0-37), all behaviors have unique IDs

    mapping = {}

    # Most IDs shift due to 'bite' removal at ID 15
    # IDs 0-14 stay the same
    for i in range(15):
        mapping[i] = i

    # V8 ID 15 was 'bite' (not in data) - no mapping

    # V8 IDs 16-27 shift down by 1 in V8.5
    v8_actions_16_27 = ['dominance', 'defend', 'flinch', 'avoid', 'escape',
                        'freeze', 'allogroom', 'shepherd', 'disengage',
                        'run', 'dominancegroom', 'huddle']

    for v8_id in range(16, 28):
        v8_5_id = v8_id - 1  # Shift down due to bite removal
        mapping[v8_id] = v8_5_id

    return mapping

def print_mapping_summary():
    """Print a summary of the action mapping"""
    print("="*80)
    print("V8.5 ACTION MAPPING SUMMARY")
    print("="*80)
    print(f"\nTotal classes: {NUM_ACTIONS} (0=background, 1-37=behaviors)")
    print(f"\nBehaviors by category:")

    for category, ids in ACTION_CATEGORIES.items():
        print(f"\n{category.upper()} ({len(ids)} behaviors):")
        for action_id in ids:
            action_name = ID_TO_ACTION[action_id]
            freq_group = get_frequency_group(action_id)
            print(f"  [{action_id:2d}] {action_name:20s} - {freq_group}")

    print(f"\n{'='*80}")
    print("KEY IMPROVEMENTS FROM V8/V8.1:")
    print("  1. +11 behaviors (from 27 to 37)")
    print("  2. Non-social behaviors now included (rear, selfgroom, etc.)")
    print("  3. Removed 'bite' (not in training data)")
    print("  4. Fully compliant with competition requirements")
    print("="*80)

if __name__ == '__main__':
    print_mapping_summary()
