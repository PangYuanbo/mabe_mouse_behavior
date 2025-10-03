"""
V6 Trainer with Dual Metrics (Frame-level + Interval-level)
Shows both training metrics and competition-style interval evaluation
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from .competition_metrics import DualMetrics
from .interval_converter import batch_frame_to_intervals, labels_to_intervals


class DualMetricTrainer:
    """
    V6 Trainer with dual evaluation:
    1. Frame-level metrics: Fast, for training monitoring
    2. Interval-level metrics: Competition-style, for true performance
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device='cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

        # Loss
        class_weights = config.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.FloatTensor(class_weights).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=config.get('min_lr', 1e-6)
        )

        # State
        self.current_epoch = 0
        self.best_val_f1 = 0.0
        self.best_interval_f1 = 0.0  # Track best competition F1
        self.patience_counter = 0

        # Checkpoint dir
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Action names (for display)
        self.action_names = config.get('action_names', ['background', 'social', 'mating', 'aggressive'])

    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')

        for batch in pbar:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # [B, T, C]

            # Loss
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = self.criterion(outputs_flat, targets_flat)

            # Backward
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Collect metrics
            _, predicted = torch.max(outputs_flat, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_flat.cpu().numpy())

            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / num_batches
        avg_acc = accuracy_score(all_targets, all_preds)
        avg_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

        return {
            'loss': avg_loss,
            'accuracy': avg_acc,
            'f1': avg_f1
        }

    def validate(self):
        """
        Validate with dual metrics:
        1. Frame-level: accuracy, F1
        2. Interval-level: competition F1, precision, recall
        """
        self.model.eval()
        epoch_loss = 0
        num_batches = 0

        # Dual metrics tracker
        dual_metrics = DualMetrics(
            num_actions=self.config['num_classes'],
            iou_threshold=self.config.get('iou_threshold', 0.5),
            min_interval_duration=self.config.get('min_interval_duration', 5)
        )

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                batch_size, seq_len = targets.shape

                # Forward
                outputs = self.model(inputs)  # [B, T, C]

                # Loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                loss = self.criterion(outputs_flat, targets_flat)
                epoch_loss += loss.item()
                num_batches += 1

                # Get predictions
                probs = torch.softmax(outputs, dim=-1)  # [B, T, C]
                _, preds = torch.max(outputs, dim=-1)  # [B, T]

                # Update frame-level metrics
                preds_np = preds.cpu().numpy()
                targets_np = targets.cpu().numpy()
                dual_metrics.update_frame_level(preds_np, targets_np)

                # Convert to intervals for interval-level metrics
                pred_intervals, gt_intervals = batch_frame_to_intervals(
                    batch_preds=preds,
                    batch_probs=probs,
                    batch_targets=None,  # Will convert from frame labels
                    min_duration=self.config.get('min_interval_duration', 5)
                )

                # Convert ground truth frame labels to intervals
                gt_intervals = []
                for b in range(batch_size):
                    gt_ivs = labels_to_intervals(targets_np[b])
                    gt_intervals.append(gt_ivs)

                # Update interval-level metrics
                dual_metrics.update_interval_level(pred_intervals, gt_intervals)

        # Compute all metrics
        all_metrics = dual_metrics.compute_all()

        avg_loss = epoch_loss / num_batches

        return {
            'loss': avg_loss,
            'frame_metrics': all_metrics['frame'],
            'interval_metrics': all_metrics['interval'],
            'interval_per_action': all_metrics['interval_per_action'],
        }

    def train(self):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"Training with Dual Metrics (Frame-level + Interval-level)")
        print(f"{'='*70}")
        print(f"Epochs: {self.config.get('epochs', 100)}")
        print(f"Learning rate: {self.config.get('learning_rate', 1e-4)}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"IoU threshold: {self.config.get('iou_threshold', 0.5)}")
        print(f"Min interval duration: {self.config.get('min_interval_duration', 5)} frames")
        print(f"{'='*70}\n")

        for epoch in range(self.config.get('epochs', 100)):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Print results
            self._print_epoch_results(train_metrics, val_metrics)

            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Save checkpoint
            is_best = self._save_checkpoint(val_metrics)

            # Early stopping
            if not is_best:
                self.patience_counter += 1
                if self.patience_counter >= self.config.get('early_stopping_patience', 15):
                    print(f"\n[X] Early stopping triggered (patience={self.config.get('early_stopping_patience', 15)})")
                    break
            else:
                self.patience_counter = 0

        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"{'='*70}")
        print(f"Best Frame-level F1:    {self.best_val_f1:.4f}")
        print(f"Best Interval-level F1: {self.best_interval_f1:.4f}")
        print(f"{'='*70}\n")

    def _print_epoch_results(self, train_metrics, val_metrics):
        """Print epoch results with both frame and interval metrics"""
        print(f"\n{'='*70}")
        print(f"Epoch {self.current_epoch + 1}/{self.config.get('epochs', 100)}")
        print(f"{'='*70}")

        # Training metrics
        print(f"[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")

        # Validation frame-level metrics
        frame_metrics = val_metrics['frame_metrics']
        print(f"\n[Val Frame-level]")
        print(f"  Loss:     {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {frame_metrics['frame_accuracy']:.4f}")

        # Validation interval-level metrics (COMPETITION METRICS)
        interval_metrics = val_metrics['interval_metrics']
        print(f"\n[Val Interval-level] ⭐ COMPETITION METRICS ⭐")
        print(f"  Precision: {interval_metrics['precision']:.4f}")
        print(f"  Recall:    {interval_metrics['recall']:.4f}")
        print(f"  F1:        {interval_metrics['f1']:.4f}")
        print(f"  TP/FP/FN:  {interval_metrics['tp']}/{interval_metrics['fp']}/{interval_metrics['fn']}")

        # Per-action interval metrics
        per_action = val_metrics['interval_per_action']
        if len(per_action) > 0:
            print(f"\n[Per-Action Interval F1]")
            for action_id, metrics in per_action.items():
                action_name = self.action_names[action_id] if action_id < len(self.action_names) else f'class_{action_id}'
                print(f"  {action_name:12s}: F1={metrics['f1']:.4f} "
                      f"(P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                      f"TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']})")

        print(f"{'='*70}")

    def _save_checkpoint(self, val_metrics):
        """Save checkpoint if best model"""
        interval_f1 = val_metrics['interval_metrics']['f1']
        frame_acc = val_metrics['frame_metrics']['frame_accuracy']

        is_best = interval_f1 > self.best_interval_f1

        if is_best:
            self.best_interval_f1 = interval_f1

            checkpoint = {
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_interval_f1': self.best_interval_f1,
                'val_metrics': val_metrics,
                'config': self.config,
            }

            save_path = self.checkpoint_dir / 'best_model_v6_dual.pth'
            torch.save(checkpoint, save_path)
            print(f"\n✓ Saved best model (Interval F1: {interval_f1:.4f}) to {save_path}")

        return is_best
