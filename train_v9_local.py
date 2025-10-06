"""
V9 Training Script - Interval Assembler
Train the boundary detection heads on top of frozen V8

Run: python train_v9_local.py --config configs/config_v9_assembler.yaml
"""

import torch
import yaml
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from versions.v9_interval_assembler.v9_dataset import create_v9_dataloaders
from versions.v9_interval_assembler.assembler_model import IntervalAssembler
from versions.v9_interval_assembler.assembler_loss import AssemblerLoss
from versions.v9_interval_assembler.decoder import decode_intervals, temporal_nms
from versions.v8_fine_grained.submission_utils import evaluate_intervals, predictions_to_intervals


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_boundary_loss = 0
    total_consistency_loss = 0
    total_fragment_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        keypoints = batch['keypoints'].to(device)
        start_labels = batch['start_labels'].to(device)
        end_labels = batch['end_labels'].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_start, pred_end, pred_conf = model(keypoints)

        # Compute loss (no interval decoding during training for speed)
        loss, loss_dict = criterion(
            pred_start, pred_end, pred_conf,
            start_labels, end_labels,
            pred_intervals=None,  # Skip slow interval decoding during training
            gt_intervals=None
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['total']
        total_boundary_loss += loss_dict['boundary']
        total_consistency_loss += loss_dict['consistency']
        total_fragment_loss += loss_dict['fragment']
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'boundary': f"{loss_dict['boundary']:.4f}"
        })

    return {
        'loss': total_loss / num_batches,
        'boundary_loss': total_boundary_loss / num_batches,
        'consistency_loss': total_consistency_loss / num_batches,
        'fragment_loss': total_fragment_loss / num_batches
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, compute_interval_f1=False):
    """Validate model"""
    model.eval()

    total_loss = 0
    total_boundary_loss = 0
    num_batches = 0

    # For interval F1 computation
    all_pred_intervals = []
    all_gt_intervals = []

    for batch in tqdm(val_loader, desc="Validation"):
        keypoints = batch['keypoints'].to(device)
        start_labels = batch['start_labels'].to(device)
        end_labels = batch['end_labels'].to(device)
        action = batch['action']
        agent = batch['agent']
        target = batch['target']

        # Forward pass
        pred_start, pred_end, pred_conf = model(keypoints)

        # Compute loss
        loss, loss_dict = criterion(
            pred_start, pred_end, pred_conf,
            start_labels, end_labels
        )

        total_loss += loss_dict['total']
        total_boundary_loss += loss_dict['boundary']
        num_batches += 1

        # Decode intervals (if requested)
        if compute_interval_f1:
            B = keypoints.shape[0]
            for b in range(B):
                # Decode predicted intervals
                pred_intervals = decode_intervals(
                    pred_start[b].cpu().numpy(),
                    pred_end[b].cpu().numpy(),
                    pred_conf[b].cpu().numpy()
                )
                pred_intervals = temporal_nms(pred_intervals, iou_threshold=0.3)
                all_pred_intervals.extend(pred_intervals)

                # Ground truth intervals
                gt_intervals = predictions_to_intervals(
                    action[b].numpy(),
                    agent[b].numpy(),
                    target[b].numpy(),
                    min_duration=1
                )
                all_gt_intervals.extend(gt_intervals)

    metrics = {
        'loss': total_loss / num_batches,
        'boundary_loss': total_boundary_loss / num_batches
    }

    # Compute interval F1
    if compute_interval_f1:
        interval_metrics = evaluate_intervals(all_pred_intervals, all_gt_intervals, iou_threshold=0.5)
        metrics['interval_f1'] = interval_metrics['f1']
        metrics['interval_precision'] = interval_metrics['precision']
        metrics['interval_recall'] = interval_metrics['recall']
        metrics['num_pred'] = len(all_pred_intervals)
        metrics['num_gt'] = len(all_gt_intervals)

    return metrics


def main():
    parser = argparse.ArgumentParser(description='V9 Interval Assembler Training')
    parser.add_argument('--config', type=str, default='configs/config_v9_assembler.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("V9 Interval Assembler Training")
    print("="*60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create dataloaders
    print(f"\nLoading data...")
    train_loader, val_loader = create_v9_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        stride=config.get('stride', 25),
        sigma=config.get('boundary_sigma', 2.0),
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )

    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")

    # Build model
    print(f"\nBuilding V9 Assembler...")
    model = IntervalAssembler(
        v8_checkpoint=config['v8_checkpoint'],
        v8_config=config['v8_config'],
        encoder_type=config.get('encoder_type', 'bilstm'),
        encoder_hidden=config.get('encoder_hidden', 256),
        encoder_layers=config.get('encoder_layers', 2),
        freeze_v8=config.get('freeze_v8', True)
    ).to(device)

    # Loss function
    criterion = AssemblerLoss(
        boundary_weight=config.get('boundary_weight', 1.0),
        iou_weight=config.get('iou_weight', 0.5),
        consistency_weight=config.get('consistency_weight', 0.1),
        fragment_weight=config.get('fragment_weight', 0.05),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0)
    )

    # Optimizer (only train assembler heads, V8 is frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[OK] Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'],
        eta_min=config.get('min_lr', 1e-6)
    )

    # Training loop
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    best_val_f1 = 0
    patience_counter = 0

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Checkpoint dir: {checkpoint_dir}\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate (compute interval F1 every 5 epochs)
        compute_f1 = (epoch % 5 == 0 or epoch == config['epochs'])
        val_metrics = validate(model, val_loader, criterion, device, compute_interval_f1=compute_f1)

        # Print metrics
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Boundary: {train_metrics['boundary_loss']:.4f} | "
              f"Consistency: {train_metrics['consistency_loss']:.4f} | "
              f"Fragment: {train_metrics['fragment_loss']:.4f}")

        if 'interval_f1' in val_metrics:
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
                  f"Boundary: {val_metrics['boundary_loss']:.4f}")
            print(f"[Kaggle] Interval F1: {val_metrics['interval_f1']:.4f} | "
                  f"Precision: {val_metrics['interval_precision']:.4f} | "
                  f"Recall: {val_metrics['interval_recall']:.4f}")
            print(f"  Predicted: {val_metrics['num_pred']} intervals | GT: {val_metrics['num_gt']} intervals")

            # Save best model
            if val_metrics['interval_f1'] > best_val_f1:
                best_val_f1 = val_metrics['interval_f1']
                torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
                print(f"\n[OK] Saved best model (val_f1={best_val_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
                  f"Boundary: {val_metrics['boundary_loss']:.4f}")

        # Early stopping
        if patience_counter >= config.get('early_stopping_patience', 20):
            print(f"\n[X] Early stopping triggered (patience={patience_counter})")
            break

        # Step scheduler
        scheduler.step()

    print(f"\n{'='*60}")
    print(f"[OK] Training complete!")
    print(f"  Best Val Interval F1: {best_val_f1:.4f}")
    print(f"  Checkpoint: {checkpoint_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
