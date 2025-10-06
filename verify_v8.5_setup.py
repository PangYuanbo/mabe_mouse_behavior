"""
V8.5 Setup Verification Script
Checks that V8.5 is correctly installed and ready for training
"""

import sys
from pathlib import Path

# Force UTF-8 encoding for Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*70)
print("V8.5 Setup Verification")
print("="*70)

all_checks_passed = True

# Check 1: Import action_mapping
print("\n[1/8] Checking action_mapping...")
try:
    from versions.v8_5_full_behaviors.action_mapping import (
        NUM_ACTIONS, ID_TO_ACTION, ACTION_TO_ID, get_action_name
    )
    assert NUM_ACTIONS == 38, f"Expected NUM_ACTIONS=38, got {NUM_ACTIONS}"
    assert len(ID_TO_ACTION) == 38, f"Expected 38 classes, got {len(ID_TO_ACTION)}"
    print(f"  ✅ NUM_ACTIONS = {NUM_ACTIONS}")
    print(f"  ✅ Total behaviors: {len(ID_TO_ACTION) - 1}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 2: Import model
print("\n[2/8] Checking v8_5_model...")
try:
    from versions.v8_5_full_behaviors.v8_5_model import (
        V85BehaviorDetector, V85MultiTaskLoss
    )
    print(f"  ✅ V85BehaviorDetector imported")
    print(f"  ✅ V85MultiTaskLoss imported")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 3: Import dataset
print("\n[3/8] Checking v8_5_dataset...")
try:
    from versions.v8_5_full_behaviors.v8_5_dataset import (
        V85Dataset, create_v85_dataloaders
    )
    print(f"  ✅ V85Dataset imported")
    print(f"  ✅ create_v85_dataloaders imported")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 4: Config file exists
print("\n[4/8] Checking config file...")
try:
    config_file = Path('configs/config_v8.5_full_behaviors.yaml')
    assert config_file.exists(), f"Config file not found: {config_file}"
    print(f"  ✅ Config file exists: {config_file}")

    # Read and check num_actions
    with open(config_file) as f:
        config_content = f.read()
        assert 'num_actions: 38' in config_content, "Config missing num_actions: 38"
    print(f"  ✅ Config has num_actions: 38")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 5: Training script exists
print("\n[5/8] Checking training script...")
try:
    train_script = Path('train_v8.5_local.py')
    assert train_script.exists(), f"Training script not found: {train_script}"
    print(f"  ✅ Training script exists: {train_script}")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 6: New behaviors included
print("\n[6/8] Checking newly included behaviors...")
try:
    # These were mapped to 0 in V8/V8.1, should have unique IDs in V8.5
    new_behaviors = {
        'rear': 31,
        'selfgroom': 27,
        'dig': 33,
        'climb': 34,
        'genitalgroom': 26,
        'dominancemount': 29,
        'tussle': 30,
        'rest': 32,
        'exploreobject': 35,
        'biteobject': 36,
        'submit': 37,
    }

    for behavior, expected_id in new_behaviors.items():
        actual_id = ACTION_TO_ID.get(behavior)
        assert actual_id == expected_id, f"{behavior}: expected ID {expected_id}, got {actual_id}"

    print(f"  ✅ All 11 new behaviors correctly mapped")
    print(f"      (rear, selfgroom, dig, climb, etc.)")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 7: 'bite' removed
print("\n[7/8] Checking 'bite' removed...")
try:
    # 'bite' should not be in ID_TO_ACTION (except mapped to 0)
    for action_id, action_name in ID_TO_ACTION.items():
        if action_id > 0:
            assert action_name != 'bite', f"'bite' still at ID {action_id}"

    # ACTION_TO_ID might have 'bite' -> 0, that's ok
    bite_id = ACTION_TO_ID.get('bite', -1)
    assert bite_id <= 0, f"'bite' has ID {bite_id}, should be removed or 0"

    print(f"  ✅ 'bite' correctly removed from behavior classes")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Check 8: Data directory
print("\n[8/8] Checking data directory...")
try:
    data_dir = Path('data/kaggle')
    assert data_dir.exists(), f"Data directory not found: {data_dir}"

    train_annotation = data_dir / 'train_annotation'
    assert train_annotation.exists(), f"Train annotation dir not found: {train_annotation}"

    # Count subdirectories (labs)
    labs = [d for d in train_annotation.iterdir() if d.is_dir()]
    print(f"  ✅ Data directory exists: {data_dir}")
    print(f"  ✅ Found {len(labs)} labs in train_annotation")
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    all_checks_passed = False

# Summary
print("\n" + "="*70)
if all_checks_passed:
    print("✅ ALL CHECKS PASSED!")
    print("\nV8.5 is ready for training. Run:")
    print("  start_v8.5_training.bat")
    print("or")
    print("  python train_v8.5_local.py --config configs/config_v8.5_full_behaviors.yaml")
else:
    print("❌ SOME CHECKS FAILED")
    print("\nPlease fix the issues above before training.")
    sys.exit(1)

print("="*70)

# Print sample behaviors
print("\nSample behaviors (first 15):")
for i in range(min(15, NUM_ACTIONS)):
    print(f"  [{i:2d}] {ID_TO_ACTION[i]}")
print(f"  ... ({NUM_ACTIONS} classes total)")
