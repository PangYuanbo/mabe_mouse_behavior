"""
Test V8 components
"""

import torch
import numpy as np
from versions.v8_fine_grained import (
    V8BehaviorDetector,
    V8MultiTaskLoss,
    ACTION_TO_ID,
    ID_TO_ACTION,
    predictions_to_intervals,
    create_submission
)

def test_model():
    """Test V8 model forward pass"""
    print("Testing V8 model...")

    model = V8BehaviorDetector(
        input_dim=288,
        num_actions=28,
        num_mice=4
    )

    # Test forward pass
    batch_size = 2
    seq_len = 100
    x = torch.randn(batch_size, seq_len, 288)

    action_logits, agent_logits, target_logits = model(x)

    assert action_logits.shape == (batch_size, seq_len, 28)
    assert agent_logits.shape == (batch_size, seq_len, 4)
    assert target_logits.shape == (batch_size, seq_len, 4)

    print("[OK] Model forward pass")
    print(f"  Output shapes: {action_logits.shape}, {agent_logits.shape}, {target_logits.shape}")

    return model

def test_loss():
    """Test V8 multi-task loss"""
    print("\nTesting V8 loss...")

    criterion = V8MultiTaskLoss(
        action_weight=1.0,
        agent_weight=0.3,
        target_weight=0.3,
        use_focal=True
    )

    batch_size = 2
    seq_len = 100

    action_logits = torch.randn(batch_size, seq_len, 28)
    agent_logits = torch.randn(batch_size, seq_len, 4)
    target_logits = torch.randn(batch_size, seq_len, 4)

    action_labels = torch.randint(0, 28, (batch_size, seq_len))
    agent_labels = torch.randint(0, 4, (batch_size, seq_len))
    target_labels = torch.randint(0, 4, (batch_size, seq_len))

    loss, loss_dict = criterion(
        action_logits, agent_logits, target_logits,
        action_labels, agent_labels, target_labels
    )

    assert isinstance(loss, torch.Tensor)
    assert 'total' in loss_dict
    assert 'action' in loss_dict
    assert 'agent' in loss_dict
    assert 'target' in loss_dict

    print("[OK] Loss computation")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    print(f"  Action loss: {loss_dict['action']:.4f}")
    print(f"  Agent loss: {loss_dict['agent']:.4f}")
    print(f"  Target loss: {loss_dict['target']:.4f}")

def test_action_mapping():
    """Test action mapping"""
    print("\nTesting action mapping...")

    # Test some mappings
    assert ACTION_TO_ID['sniff'] == 1
    assert ACTION_TO_ID['mount'] == 8
    assert ACTION_TO_ID['attack'] == 12
    assert ACTION_TO_ID['background'] == 0

    assert ID_TO_ACTION[1] == 'sniff'
    assert ID_TO_ACTION[8] == 'mount'
    assert ID_TO_ACTION[12] == 'attack'
    assert ID_TO_ACTION[0] == 'background'

    print("[OK] Action mapping")
    print(f"  Total actions: {len(ID_TO_ACTION)}")

def test_interval_conversion():
    """Test predictions to intervals conversion"""
    print("\nTesting interval conversion...")

    T = 200
    # Create fake predictions
    action_preds = np.zeros((T, 28))
    action_preds[:, 0] = 0.9  # Mostly background

    # Add some behaviors
    action_preds[50:70, 1] = 1.0  # sniff
    action_preds[100:130, 8] = 1.0  # mount

    agent_preds = np.zeros((T, 4))
    agent_preds[:, 0] = 1.0  # Mouse 1

    target_preds = np.zeros((T, 4))
    target_preds[:, 1] = 1.0  # Mouse 2

    intervals = predictions_to_intervals(
        action_preds=action_preds,
        agent_preds=agent_preds,
        target_preds=target_preds,
        min_duration=5
    )

    print(f"[OK] Interval conversion")
    print(f"  Found {len(intervals)} intervals")
    for interval in intervals[:3]:
        print(f"    {interval['action']}: frames {interval['start_frame']}-{interval['stop_frame']}")

def test_submission():
    """Test submission creation"""
    print("\nTesting submission creation...")

    # Create fake predictions for 2 videos
    predictions = {}

    for i, video_id in enumerate(['video1', 'video2']):
        T = 150
        action_preds = np.zeros((T, 28))
        action_preds[:, 0] = 0.9
        action_preds[20:40, 1] = 1.0  # sniff

        agent_preds = np.zeros((T, 4))
        agent_preds[:, i % 4] = 1.0

        target_preds = np.zeros((T, 4))
        target_preds[:, (i + 1) % 4] = 1.0

        predictions[video_id] = {
            'action': action_preds,
            'agent': agent_preds,
            'target': target_preds
        }

    submission = create_submission(
        predictions=predictions,
        video_ids=['video1', 'video2'],
        min_duration=5
    )

    print(f"[OK] Submission creation")
    print(f"  Total predictions: {len(submission)}")
    print(f"  Columns: {submission.columns.tolist()}")
    if len(submission) > 0:
        print(f"\n  Sample submission:")
        print(submission.head(3).to_string())

def main():
    print("="*60)
    print("V8 Component Tests")
    print("="*60)

    test_action_mapping()
    test_model()
    test_loss()
    test_interval_conversion()
    test_submission()

    print("\n" + "="*60)
    print("[OK] All tests passed!")
    print("="*60)

if __name__ == '__main__':
    main()
