"""
V8.6 Training Script - MARS-Enhanced with Freeze Detection
===========================================================

Key improvements over V8.5:
1. MARS-inspired features (~370 dims)
2. Multi-scale temporal modeling
3. Dedicated Freeze detection branch
4. Dynamic class weighting (3x for rare behaviors)
5. Rare behavior oversampling (3x)
6. Longer sequences (150 frames = 4.5 sec)
7. Self-attention mechanism

Run: python train_v8_6_local.py --config configs/config_v8.6_mars.yaml
"""

import torch
import yaml
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from versions.v8_6_mars_enhanced.v8_6_dataset import create_v86_dataloaders
from versions.v8_6_mars_enhanced.v8_6_model import V86BehaviorDetector, V86MultiTaskLoss
from versions.v8_6_mars_enhanced.action_mapping import NUM_ACTIONS, ID_TO_ACTION, FREQUENCY_GROUPS
from versions.v8_fine_grained.submission_utils import predictions_to_intervals, evaluate_intervals
from src.utils.detailed_metrics import (
    compute_per_class_metrics,
    compute_interval_per_class_f1,
    print_detailed_metrics
)


def compute_dynamic_class_weights(train_loader, num_classes, rare_boost=3.0, device='cuda'):
    """
    Compute class weights with boosted weights for rare behaviors

    Args:
        train_loader: DataLoader
        num_classes: Number of classes (38)
        rare_boost: Multiplier for rare behavior weights (default 3.0)
        device: torch device

    Returns:
        class_weights: [num_classes] tensor
        freeze_class_weights: [2] tensor for freeze binary classification
    """
    print(f"  Computing dynamic class weights (rare_boost={rare_boost})...")

    # Count occurrences of each class
    class_counts = np.zeros(num_classes, dtype=np.int64)
    freeze_counts = np.zeros(2, dtype=np.int64)  # [non-freeze, freeze]

    for keypoints, action, agent, target, freeze in tqdm(train_loader, desc="Counting classes"):
        action_np = action.cpu().numpy().flatten()
        freeze_np = freeze.cpu().numpy().flatten()

        for class_id in range(num_classes):
            class_counts[class_id] += (action_np == class_id).sum()

        freeze_counts[0] += (freeze_np == 0).sum()  # Non-freeze
        freeze_counts[1] += (freeze_np == 1).sum()  # Freeze

    total = class_counts.sum()

    print(f"\n  Class distribution (top 15):")
    sorted_idx = np.argsort(-class_counts)[:15]
    for rank, class_id in enumerate(sorted_idx):
        action_name = ID_TO_ACTION.get(class_id, f'class_{class_id}')
        pct = class_counts[class_id] / total * 100
        print(f"    {rank+1:2d}. [{class_id:2d}] {action_name:20s}: {class_counts[class_id]:10,} ({pct:5.2f}%)")

    # Compute base weights (inverse sqrt frequency)
    weights = np.sqrt(total / (class_counts + 1))

    # Boost rare behaviors
    rare_behavior_ids = [
        20,  # freeze
        19,  # escape
        25,  # dominancegroom
        10,  # attemptmount
        11,  # ejaculate
        17,  # flinch
        23,  # disengage
        21,  # allogroom
        36,  # biteobject
        35,  # exploreobject
        37,  # submit
    ]

    print(f"\n  Boosting rare behaviors ({rare_boost}x):")
    for rare_id in rare_behavior_ids:
        if rare_id < num_classes:
            action_name = ID_TO_ACTION.get(rare_id, f'class_{rare_id}')
            old_weight = weights[rare_id]
            weights[rare_id] *= rare_boost
            print(f"    [{rare_id:2d}] {action_name:20s}: {old_weight:.2f} -> {weights[rare_id]:.2f}")

    # Normalize weights
    weights = weights / weights.mean()

    # Freeze binary weights
    freeze_total = freeze_counts.sum()
    freeze_weights = np.sqrt(freeze_total / (freeze_counts + 1))
    freeze_weights = freeze_weights / freeze_weights.mean()

    print(f"\n  Freeze distribution:")
    print(f"    Non-freeze: {freeze_counts[0]:,} ({freeze_counts[0]/freeze_total*100:.2f}%)")
    print(f"    Freeze:     {freeze_counts[1]:,} ({freeze_counts[1]/freeze_total*100:.2f}%)")
    print(f"    Freeze weight: {freeze_weights[1]:.2f}")

    # Convert to tensors
    class_weights = torch.FloatTensor(weights).to(device)
    freeze_class_weights = torch.FloatTensor(freeze_weights).to(device)

    print(f"\n  Action weight range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")
    print(f"  Action weight mean: {class_weights.mean():.2f}")

    return class_weights, freeze_class_weights


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch with freeze detection"""
    model.train()

    total_loss = 0
    total_action_loss = 0
    total_agent_loss = 0
    total_target_loss = 0
    total_freeze_loss = 0

    action_correct = 0
    agent_correct = 0
    target_correct = 0
    freeze_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, (keypoints, action, agent, target, freeze) in enumerate(pbar):
        keypoints = keypoints.to(device)
        action = action.to(device)
        agent = agent.to(device)
        target = target.to(device)
        freeze = freeze.to(device)

        optimizer.zero_grad()

        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                action_logits, agent_logits, target_logits, freeze_logits = model(keypoints)
                loss, loss_dict = criterion(
                    action_logits, agent_logits, target_logits, freeze_logits,
                    action, agent, target, freeze
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            action_logits, agent_logits, target_logits, freeze_logits = model(keypoints)
            loss, loss_dict = criterion(
                action_logits, agent_logits, target_logits, freeze_logits,
                action, agent, target, freeze
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Compute accuracies
        action_pred = action_logits.argmax(dim=-1)
        agent_pred = agent_logits.argmax(dim=-1)
        target_pred = target_logits.argmax(dim=-1)
        freeze_pred = freeze_logits.argmax(dim=-1)

        action_correct += (action_pred == action).sum().item()
        agent_correct += (agent_pred == agent).sum().item()
        target_correct += (target_pred == target).sum().item()
        freeze_correct += (freeze_pred == freeze).sum().item()

        batch_samples = action.numel()
        total_samples += batch_samples

        total_loss += loss_dict['total'] * batch_samples
        total_action_loss += loss_dict['action'] * batch_samples
        total_agent_loss += loss_dict['agent'] * batch_samples
        total_target_loss += loss_dict['target'] * batch_samples
        total_freeze_loss += loss_dict['freeze'] * batch_samples

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'act_acc': f"{action_correct / total_samples:.4f}",
            'agt_acc': f"{agent_correct / total_samples:.4f}",
            'frz_acc': f"{freeze_correct / total_samples:.4f}"
        })

    # Average metrics
    metrics = {
        'loss': total_loss / total_samples,
        'action_loss': total_action_loss / total_samples,
        'agent_loss': total_agent_loss / total_samples,
        'target_loss': total_target_loss / total_samples,
        'freeze_loss': total_freeze_loss / total_samples,
        'action_acc': action_correct / total_samples,
        'agent_acc': agent_correct / total_samples,
        'target_acc': target_correct / total_samples,
        'freeze_acc': freeze_correct / total_samples,
    }

    return metrics


@torch.no_grad()
def validate_epoch(model, val_loader, device, num_actions=38):
    """Validate model with detailed metrics"""
    model.eval()

    all_action_preds = []
    all_action_labels = []
    all_agent_preds = []
    all_agent_labels = []
    all_target_preds = []
    all_target_labels = []
    all_freeze_preds = []
    all_freeze_labels = []

    pbar = tqdm(val_loader, desc="Validation")

    for keypoints, action, agent, target, freeze in pbar:
        keypoints = keypoints.to(device)

        # Forward pass
        action_logits, agent_logits, target_logits, freeze_logits = model(keypoints)

        # Predictions
        action_pred = action_logits.argmax(dim=-1).cpu().numpy()
        agent_pred = agent_logits.argmax(dim=-1).cpu().numpy()
        target_pred = target_logits.argmax(dim=-1).cpu().numpy()
        freeze_pred = freeze_logits.argmax(dim=-1).cpu().numpy()

        all_action_preds.append(action_pred)
        all_action_labels.append(action.numpy())
        all_agent_preds.append(agent_pred)
        all_agent_labels.append(agent.numpy())
        all_target_preds.append(target_pred)
        all_target_labels.append(target.numpy())
        all_freeze_preds.append(freeze_pred)
        all_freeze_labels.append(freeze.numpy())

    # Concatenate all batches
    action_preds = np.concatenate(all_action_preds, axis=0).flatten()
    action_labels = np.concatenate(all_action_labels, axis=0).flatten()
    agent_preds = np.concatenate(all_agent_preds, axis=0).flatten()
    agent_labels = np.concatenate(all_agent_labels, axis=0).flatten()
    target_preds = np.concatenate(all_target_preds, axis=0).flatten()
    target_labels = np.concatenate(all_target_labels, axis=0).flatten()
    freeze_preds = np.concatenate(all_freeze_preds, axis=0).flatten()
    freeze_labels = np.concatenate(all_freeze_labels, axis=0).flatten()

    # Compute metrics
    action_acc = (action_preds == action_labels).mean()
    agent_acc = (agent_preds == agent_labels).mean()
    target_acc = (target_preds == target_labels).mean()
    freeze_acc = (freeze_preds == freeze_labels).mean()

    # Per-class metrics
    per_class_metrics = compute_per_class_metrics(
        action_labels, action_preds, num_classes=num_actions
    )

    # Compute interval F1 (Kaggle metric)
    interval_f1, precision, recall = compute_interval_per_class_f1(
        action_labels, action_preds, num_classes=num_actions
    )

    metrics = {
        'action_acc': action_acc,
        'agent_acc': agent_acc,
        'target_acc': target_acc,
        'freeze_acc': freeze_acc,
        'interval_f1': interval_f1,
        'precision': precision,
        'recall': recall,
        'per_class': per_class_metrics
    }

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_v8.6_mars.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("V8.6 MARS-Enhanced Training")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Mixed Precision: {config['training']['use_mixed_precision']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Sequence Length: {config['dataset']['sequence_length']}")
    print(f"Rare Behavior Oversampling: {config['dataset'].get('oversample_rare', True)}")
    print()

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_v86_dataloaders(
        data_dir=config['dataset']['data_dir'],
        batch_size=config['training']['batch_size'],
        sequence_length=config['dataset']['sequence_length'],
        stride=config['dataset'].get('stride', 37),
        num_workers=config['dataset'].get('num_workers', 0),
        pin_memory=True,
        oversample_rare=config['dataset'].get('oversample_rare', True),
        augment=config['dataset'].get('augment', True)
    )

    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")
    print()

    # Compute class weights
    class_weights, freeze_class_weights = compute_dynamic_class_weights(
        train_loader,
        num_classes=NUM_ACTIONS,
        rare_boost=config['training'].get('rare_boost', 3.0),
        device=device
    )

    # Create model
    print("\nCreating V8.6 model...")
    model = V86BehaviorDetector(
        input_dim=config['model'].get('input_dim', 370),
        num_actions=NUM_ACTIONS,
        conv_channels=config['model'].get('conv_channels', [192, 384, 512]),
        lstm_hidden=config['model'].get('lstm_hidden', 384),
        lstm_layers=config['model'].get('lstm_layers', 3),
        dropout=config['model'].get('dropout', 0.3),
        use_attention=config['model'].get('use_attention', True)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Model parameters: {total_params:,}")
    print()

    # Loss criterion
    criterion = V86MultiTaskLoss(
        action_weight=config['training'].get('action_weight', 1.0),
        agent_weight=config['training'].get('agent_weight', 0.3),
        target_weight=config['training'].get('target_weight', 0.3),
        freeze_weight=config['training'].get('freeze_weight', 0.5),
        use_focal=config['training'].get('use_focal', True),
        focal_gamma=config['training'].get('focal_gamma', 2.0),
        class_weights=class_weights,
        freeze_class_weights=freeze_class_weights
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['learning_rate'] * 0.01
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['training']['use_mixed_precision'] else None

    # Training loop
    print("=" * 60)
    print(f"Training for {config['training']['num_epochs']} epochs")
    print("=" * 60)
    print()

    best_interval_f1 = 0.0
    patience = config['training'].get('early_stopping_patience', 15)
    patience_counter = 0

    for epoch in range(1, config['training']['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['training']['num_epochs']}")
        print('='*60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate
        val_metrics = validate_epoch(model, val_loader, device, num_actions=NUM_ACTIONS)

        # Print results
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Action Acc: {train_metrics['action_acc']:.4f} | "
              f"Agent Acc: {train_metrics['agent_acc']:.4f} | "
              f"Freeze Acc: {train_metrics['freeze_acc']:.4f}")

        print(f"[Val]   Action Acc: {val_metrics['action_acc']:.4f} | "
              f"Agent Acc: {val_metrics['agent_acc']:.4f} | "
              f"Freeze Acc: {val_metrics['freeze_acc']:.4f}")

        print(f"[Kaggle] Interval F1: {val_metrics['interval_f1']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f}")

        # Detailed metrics
        print_detailed_metrics(val_metrics['per_class'], ID_TO_ACTION)

        # Learning rate step
        scheduler.step()

        # Save best model
        if val_metrics['interval_f1'] > best_interval_f1:
            best_interval_f1 = val_metrics['interval_f1']
            patience_counter = 0

            save_path = Path(config['training']['checkpoint_dir']) / 'v8_6_best.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'interval_f1': best_interval_f1,
                'config': config
            }, save_path)

            print(f"\n✓ Saved best model (Interval F1: {best_interval_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered (patience={patience})")
            break

    print("\n" + "=" * 60)
    print(f"Training completed! Best Interval F1: {best_interval_f1:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
