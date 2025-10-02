"""
V7 Interval Detection - Local Training (RTX 5090)
Run from project root: python train_v7_local.py
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from versions.v7_interval_detection.interval_dataset import IntervalDetectionDataset, collate_interval_fn
from versions.v7_interval_detection.interval_model import TemporalActionDetector
from versions.v7_interval_detection.interval_loss import IntervalDetectionLoss
from versions.v7_interval_detection.interval_metrics import IntervalMetrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    epoch_losses = {
        'total_loss': 0,
        'objectness_loss': 0,
        'action_loss': 0,
        'agent_loss': 0,
        'target_loss': 0,
        'boundary_loss': 0,
    }

    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for sequences, targets in pbar:
        sequences = sequences.to(device)

        # Forward pass
        predictions = model(sequences)

        # Generate anchors
        anchors = model.generate_anchors(model.sequence_length).to(device)

        # Compute loss
        losses = criterion(predictions, targets, anchors)

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()

        num_batches += 1

        # Update progress bar
        pbar.set_postfix({'loss': losses['total_loss'].item()})

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    return epoch_losses


def evaluate(model, val_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    epoch_losses = {
        'total_loss': 0,
        'objectness_loss': 0,
        'action_loss': 0,
        'agent_loss': 0,
        'target_loss': 0,
        'boundary_loss': 0,
    }

    metrics = IntervalMetrics(iou_threshold=0.5)
    num_batches = 0

    with torch.no_grad():
        for sequences, targets in tqdm(val_loader, desc='Validation'):
            sequences = sequences.to(device)

            # Forward pass
            predictions = model(sequences)

            # Generate anchors
            anchors = model.generate_anchors(model.sequence_length).to(device)

            # Compute loss
            losses = criterion(predictions, targets, anchors)

            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()

            # Get interval predictions
            pred_intervals = model.predict_intervals(
                predictions,
                score_threshold=0.5,
                nms_threshold=0.3
            )

            # Update metrics
            metrics.update(pred_intervals, targets)
            num_batches += 1

    # Average losses
    for key in epoch_losses:
        epoch_losses[key] /= num_batches

    # Compute metrics
    eval_metrics = metrics.compute()
    per_action_metrics = metrics.compute_per_action()

    return epoch_losses, eval_metrics, per_action_metrics


def main():
    parser = argparse.ArgumentParser(description='V7 Interval Detection - Local Training')
    parser.add_argument('--config', type=str, default='configs/config_v7_5090.yaml',
                       help='Path to config file')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"V7 Interval Detection Training - RTX 5090")
    print(f"{'='*60}")
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create datasets
    print(f"\nLoading datasets from {config['data_dir']}...")
    train_dataset = IntervalDetectionDataset(
        data_dir=config['data_dir'],
        split='train',
        sequence_length=config['sequence_length'],
    )

    val_dataset = IntervalDetectionDataset(
        data_dir=config['data_dir'],
        split='val',
        sequence_length=config['sequence_length'],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_interval_fn,
        pin_memory=config.get('pin_memory', True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_interval_fn,
        pin_memory=config.get('pin_memory', True),
    )

    print(f"✓ Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"✓ Val: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Create model
    print(f"\nCreating TemporalActionDetector...")
    model = TemporalActionDetector(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_actions=config['num_actions'],
        num_agents=config['num_agents'],
        sequence_length=config['sequence_length'],
        anchor_scales=config['anchor_scales'],
        iou_threshold=config.get('iou_eval_threshold', 0.5),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = IntervalDetectionLoss(
        iou_threshold_pos=config['iou_threshold_pos'],
        iou_threshold_neg=config['iou_threshold_neg'],
        alpha=config['focal_alpha'],
        gamma=config['focal_gamma'],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Training loop
    best_f1 = 0
    patience_counter = 0
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training for {config['epochs']} epochs")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"{'='*60}\n")

    for epoch in range(config['epochs']):
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)

        # Validate
        val_losses, val_metrics, per_action_metrics = evaluate(model, val_loader, criterion, device)

        # Print results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")
        print(f"Train Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_losses['total_loss']:.4f}")
        print(f"\nValidation Metrics:")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"  F1:        {val_metrics['f1']:.4f}")

        print(f"\nPer-action F1:")
        for action, metrics in per_action_metrics.items():
            print(f"  {action:12s}: {metrics['f1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")

        # Save checkpoint
        is_best = val_metrics['f1'] > best_f1

        if is_best:
            best_f1 = val_metrics['f1']
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'per_action_metrics': per_action_metrics,
                'config': config,
            }

            torch.save(checkpoint, checkpoint_dir / 'best_model_v7.pth')
            print(f"\n✓ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement for {patience_counter} epochs")

        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            print(f"\n✗ Early stopping triggered (patience={config['early_stopping_patience']})")
            break

    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"  Best F1: {best_f1:.4f}")
    print(f"  Checkpoint: {checkpoint_dir / 'best_model_v7.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
