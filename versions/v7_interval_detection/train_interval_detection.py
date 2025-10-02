"""
V7: Training script for interval detection
Direct behavior interval prediction instead of frame-level classification
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from versions.v7_interval_detection.interval_dataset import IntervalDetectionDataset, collate_interval_fn
from versions.v7_interval_detection.interval_model import TemporalActionDetector
from versions.v7_interval_detection.interval_loss import IntervalDetectionLoss
from versions.v7_interval_detection.interval_metrics import IntervalMetrics


def train_epoch(model, train_loader, criterion, optimizer, device):
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

    for sequences, targets in tqdm(train_loader, desc='Training'):
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
        optimizer.step()

        # Accumulate losses
        for key in epoch_losses:
            epoch_losses[key] += losses[key].item()

        num_batches += 1

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


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    print("Loading datasets...")
    train_dataset = IntervalDetectionDataset(
        data_dir=args.data_dir,
        split='train',
        sequence_length=args.sequence_length,
    )

    val_dataset = IntervalDetectionDataset(
        data_dir=args.data_dir,
        split='val',
        sequence_length=args.sequence_length,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_interval_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_interval_fn,
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")

    # Create model
    print("Creating model...")
    model = TemporalActionDetector(
        input_dim=142,  # Keypoints
        hidden_dim=256,
        num_actions=4,
        num_agents=4,
        sequence_length=args.sequence_length,
        anchor_scales=[10, 30, 60, 120, 240],
        iou_threshold=0.5,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = IntervalDetectionLoss(
        iou_threshold_pos=0.5,
        iou_threshold_neg=0.4,
        alpha=0.25,
        gamma=2.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_f1 = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining for {args.epochs} epochs...")
    print("=" * 60)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_losses, val_metrics, per_action_metrics = evaluate(model, val_loader, criterion, device)

        # Print results
        print(f"\nTrain Loss: {train_losses['total_loss']:.4f}")
        print(f"Val Loss: {val_losses['total_loss']:.4f}")
        print(f"\nValidation Metrics:")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall: {val_metrics['recall']:.4f}")
        print(f"  F1: {val_metrics['f1']:.4f}")

        print(f"\nPer-action F1:")
        for action, metrics in per_action_metrics.items():
            print(f"  {action}: {metrics['f1']:.4f}")

        # Save checkpoint
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'per_action_metrics': per_action_metrics,
            }

            torch.save(checkpoint, checkpoint_dir / 'best_model_v7.pth')
            print(f"\n✓ Saved best model (F1: {best_f1:.4f})")

    print("\n" + "=" * 60)
    print(f"Training complete! Best F1: {best_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train V7 Interval Detection Model')

    # Data
    parser.add_argument('--data_dir', type=str, default='/vol/data/kaggle',
                       help='Path to data directory')
    parser.add_argument('--sequence_length', type=int, default=1000,
                       help='Sequence length for training')

    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/v7',
                       help='Checkpoint directory')

    args = parser.parse_args()
    main(args)
