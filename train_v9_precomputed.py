"""
V9 Training Script - Using Precomputed V8 Features
Much faster than computing V8 on-the-fly

Run:
1. python precompute_v8_features.py  # One-time precomputation
2. python train_v9_precomputed.py --config configs/config_v9_assembler.yaml
"""

import torch
import torch.nn as nn
import yaml
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from versions.v9_interval_assembler.v9_precomputed_dataset import create_v9_precomputed_dataloaders
from versions.v9_interval_assembler.assembler_loss import AssemblerLoss
from versions.v9_interval_assembler.decoder import decode_intervals, temporal_nms
from versions.v8_fine_grained.submission_utils import evaluate_intervals, predictions_to_intervals


class V9AssemblerLite(nn.Module):
    """
    V9 Assembler without V8 (uses precomputed features)

    Input: V8 logits [B, T, 36] (already computed)
    Output: Start/End/Confidence heatmaps [B, T, 336]
    """

    def __init__(
        self,
        encoder_type='bilstm',
        encoder_hidden=256,
        encoder_layers=2,
        num_actions=28,
        num_pairs=12
    ):
        super().__init__()

        self.num_channels = num_actions * num_pairs  # 336
        v8_output_dim = 36  # 28 + 4 + 4

        # Temporal encoder
        if encoder_type == 'bilstm':
            self.temporal_encoder = nn.LSTM(
                input_size=v8_output_dim,
                hidden_size=encoder_hidden,
                num_layers=encoder_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.2 if encoder_layers > 1 else 0.0
            )
            encoder_out_dim = encoder_hidden * 2
        else:
            raise NotImplementedError(f"Encoder type {encoder_type} not implemented")

        # Detection heads
        self.start_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden, self.num_channels),
            nn.Sigmoid()
        )

        self.end_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden, self.num_channels),
            nn.Sigmoid()
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(encoder_out_dim, encoder_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(encoder_hidden, self.num_channels),
            nn.Sigmoid()
        )

        print(f"[V9 Lite] Assembler initialized:")
        print(f"  - Encoder: {encoder_type} (hidden={encoder_hidden}, layers={encoder_layers})")
        print(f"  - Output channels: {self.num_channels}")
        print(f"  - Trainable parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(self, v8_logits):
        """
        Args:
            v8_logits: [B, T, 36] Precomputed V8 outputs

        Returns:
            start_heatmap: [B, T, 336]
            end_heatmap: [B, T, 336]
            confidence: [B, T, 336]
        """
        # Temporal encoding
        encoded, _ = self.temporal_encoder(v8_logits)

        # Multi-head outputs
        start_heatmap = self.start_head(encoded)
        end_heatmap = self.end_head(encoded)
        confidence = self.confidence_head(encoded)

        return start_heatmap, end_heatmap, confidence


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_boundary_loss = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch in pbar:
        v8_logits = batch['v8_logits'].to(device)
        start_labels = batch['start_labels'].to(device)
        end_labels = batch['end_labels'].to(device)

        # DEBUG: Check label statistics (first batch only)
        if num_batches == 0:
            num_positive_start = (start_labels > 0.1).sum().item()
            num_positive_end = (end_labels > 0.1).sum().item()
            total_elements = start_labels.numel()
            print(f"\n[DEBUG] First batch label stats:")
            print(f"  Start labels > 0.1: {num_positive_start}/{total_elements} ({100*num_positive_start/total_elements:.2f}%)")
            print(f"  End labels > 0.1: {num_positive_end}/{total_elements} ({100*num_positive_end/total_elements:.2f}%)")
            print(f"  Start max: {start_labels.max().item():.4f}, mean: {start_labels.mean().item():.6f}")
            print(f"  End max: {end_labels.max().item():.4f}, mean: {end_labels.mean().item():.6f}")

        optimizer.zero_grad()

        # Forward pass (NO V8 inference needed!)
        pred_start, pred_end, pred_conf = model(v8_logits)

        # Compute loss
        loss, loss_dict = criterion(
            pred_start, pred_end, pred_conf,
            start_labels, end_labels
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_dict['total']
        total_boundary_loss += loss_dict['boundary']
        num_batches += 1

        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'boundary': f"{loss_dict['boundary']:.4f}",
            'consistency': f"{loss_dict['consistency']:.4f}",
            'fragment': f"{loss_dict['fragment']:.4f}"
        })

    return {
        'loss': total_loss / num_batches,
        'boundary_loss': total_boundary_loss / num_batches
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, compute_interval_f1=False):
    """Validate model"""
    model.eval()

    total_loss = 0
    num_batches = 0

    all_pred_intervals = []
    all_gt_intervals = []

    for batch in tqdm(val_loader, desc="Validation"):
        v8_logits = batch['v8_logits'].to(device)
        start_labels = batch['start_labels'].to(device)
        end_labels = batch['end_labels'].to(device)
        action = batch['action']
        agent = batch['agent']
        target = batch['target']

        # Forward pass
        pred_start, pred_end, pred_conf = model(v8_logits)

        # Compute loss
        loss, loss_dict = criterion(
            pred_start, pred_end, pred_conf,
            start_labels, end_labels
        )

        total_loss += loss_dict['total']
        num_batches += 1

        # Decode intervals
        if compute_interval_f1:
            B = v8_logits.shape[0]
            for b in range(B):
                pred_intervals = decode_intervals(
                    pred_start[b].cpu().numpy(),
                    pred_end[b].cpu().numpy(),
                    pred_conf[b].cpu().numpy()
                )
                pred_intervals = temporal_nms(pred_intervals, iou_threshold=0.3)
                all_pred_intervals.extend(pred_intervals)

                gt_intervals = predictions_to_intervals(
                    action[b].numpy(),
                    agent[b].numpy(),
                    target[b].numpy(),
                    min_duration=1
                )
                all_gt_intervals.extend(gt_intervals)

    metrics = {'loss': total_loss / num_batches}

    if compute_interval_f1:
        interval_metrics = evaluate_intervals(all_pred_intervals, all_gt_intervals, iou_threshold=0.5)
        metrics.update({
            'interval_f1': interval_metrics['f1'],
            'interval_precision': interval_metrics['precision'],
            'interval_recall': interval_metrics['recall'],
            'num_pred': len(all_pred_intervals),
            'num_gt': len(all_gt_intervals)
        })

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_v9_assembler.yaml')
    parser.add_argument('--train_h5', type=str, default='data/v8_features/train_v8_features.h5')
    parser.add_argument('--val_h5', type=str, default='data/v8_features/val_v8_features.h5')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("V9 Training (Precomputed V8 Features)")
    print("="*60)
    print(f"Device: {device}")
    print(f"Train features: {args.train_h5}")
    print(f"Val features: {args.val_h5}")

    # Create dataloaders
    print(f"\nLoading precomputed features...")
    train_loader, val_loader = create_v9_precomputed_dataloaders(
        train_h5=args.train_h5,
        val_h5=args.val_h5,
        batch_size=config['batch_size'],
        sigma=config.get('boundary_sigma', 2.0),
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )

    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")

    # Build model (NO V8!)
    print(f"\nBuilding V9 Assembler (Lite)...")
    model = V9AssemblerLite(
        encoder_type=config.get('encoder_type', 'bilstm'),
        encoder_hidden=config.get('encoder_hidden', 256),
        encoder_layers=config.get('encoder_layers', 2)
    ).to(device)

    # Loss
    criterion = AssemblerLoss(
        boundary_weight=config.get('boundary_weight', 1.0),
        iou_weight=config.get('iou_weight', 0.5),
        consistency_weight=config.get('consistency_weight', 0.1),
        fragment_weight=config.get('fragment_weight', 0.05)
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    print(f"\nStarting training for {config['epochs']} epochs...\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        compute_f1 = (epoch % 5 == 0 or epoch == config['epochs'])
        val_metrics = validate(model, val_loader, criterion, device, compute_interval_f1=compute_f1)

        # Print metrics
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f}")

        if 'interval_f1' in val_metrics:
            print(f"[Val]   Loss: {val_metrics['loss']:.4f}")
            print(f"[Kaggle] Interval F1: {val_metrics['interval_f1']:.4f} | "
                  f"Precision: {val_metrics['interval_precision']:.4f} | "
                  f"Recall: {val_metrics['interval_recall']:.4f}")
            print(f"  Predicted: {val_metrics['num_pred']} | GT: {val_metrics['num_gt']}")

            if val_metrics['interval_f1'] > best_val_f1:
                best_val_f1 = val_metrics['interval_f1']
                torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
                print(f"\n[OK] Saved best model (F1={best_val_f1:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            print(f"[Val]   Loss: {val_metrics['loss']:.4f}")

        # Early stopping
        if patience_counter >= config.get('early_stopping_patience', 15):
            print(f"\n[X] Early stopping")
            break

        scheduler.step()

    print(f"\n{'='*60}")
    print(f"[OK] Training complete! Best F1: {best_val_f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
