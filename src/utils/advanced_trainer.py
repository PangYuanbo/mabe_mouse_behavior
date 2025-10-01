"""
Advanced trainer with improved training strategies
Based on winning solutions and best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """
        Args:
            pred: [N, C] predictions
            target: [N] targets
        """
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
        one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_class
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=0.2):
    """
    Mixup data augmentation

    Args:
        x: Input data [batch, ...]
        y: Labels [batch, ...]
        alpha: Mixup hyperparameter

    Returns:
        mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class AdvancedTrainer:
    """Advanced trainer with improved strategies"""

    def __init__(self, model, train_loader, val_loader, config, device='cuda', epoch_callback=None):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
            epoch_callback: Optional callback function called after each epoch (e.g., for volume.commit())
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.epoch_callback = epoch_callback

        # Setup optimizer
        self.optimizer = self._build_optimizer()

        # Setup loss function
        self.criterion = self._build_criterion()

        # Setup scheduler
        self.scheduler = self._build_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_f1 = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.patience_counter = 0

        # Mixup
        self.use_mixup = config.get('mixup_alpha', 0) > 0
        self.mixup_alpha = config.get('mixup_alpha', 0.2)

        # Create checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def _build_optimizer(self):
        """Build optimizer"""
        optimizer_type = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-5)

        if optimizer_type == 'adam':
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        return optimizer

    def _build_criterion(self):
        """Build loss function"""
        loss_type = self.config.get('loss', 'cross_entropy')

        if loss_type == 'cross_entropy':
            # Check for class weights
            class_weights = self.config.get('class_weights', None)
            if class_weights is not None:
                class_weights = torch.FloatTensor(class_weights).to(self.device)

            # Check for label smoothing
            label_smoothing = self.config.get('label_smoothing', 0.0)
            if label_smoothing > 0:
                criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            else:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
        elif loss_type == 'bce':
            criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return criterion

    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.get('scheduler_factor', 0.5),
                patience=self.config.get('scheduler_patience', 5),
                min_lr=self.config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')

        for batch in pbar:
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
            else:
                # No labels available
                continue

            # Flatten input if needed
            batch_size, seq_len, *input_shape = inputs.shape
            if len(input_shape) > 1:
                inputs = inputs.reshape(batch_size, seq_len, -1)

            # Apply mixup if enabled
            if self.use_mixup:
                # Flatten for mixup
                inputs_flat = inputs.reshape(batch_size * seq_len, -1)
                targets_flat = targets.reshape(batch_size * seq_len)

                inputs_mixed, targets_a, targets_b, lam = mixup_data(
                    inputs_flat, targets_flat, self.mixup_alpha
                )

                # Reshape back
                inputs_mixed = inputs_mixed.reshape(batch_size, seq_len, -1)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs_mixed)

                # Compute mixup loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                loss = mixup_criterion(self.criterion, outputs_flat, targets_a, targets_b, lam)
            else:
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)

                # Compute loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                loss = self.criterion(outputs_flat, targets_flat)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )

            self.optimizer.step()

            # Collect metrics (without mixup)
            if not self.use_mixup:
                _, predicted = torch.max(outputs_flat, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets_flat.cpu().numpy())

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # Compute accuracy (if not using mixup)
        if not self.use_mixup and len(all_preds) > 0:
            avg_acc = accuracy_score(all_targets, all_preds)
        else:
            avg_acc = 0.0

        return avg_loss, avg_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0
        all_preds = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                if len(batch) == 2:
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                else:
                    continue

                # Flatten input if needed
                batch_size, seq_len, *input_shape = inputs.shape
                if len(input_shape) > 1:
                    inputs = inputs.reshape(batch_size, seq_len, -1)

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                outputs_flat = outputs.reshape(-1, outputs.size(-1))
                targets_flat = targets.reshape(-1)
                loss = self.criterion(outputs_flat, targets_flat)

                # Collect predictions
                _, predicted = torch.max(outputs_flat, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets_flat.cpu().numpy())

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0

        # Compute metrics
        metrics = {}
        if len(all_preds) > 0:
            metrics['accuracy'] = accuracy_score(all_targets, all_preds)
            metrics['f1_macro'] = f1_score(all_targets, all_preds, average='macro', zero_division=0)
            metrics['f1_weighted'] = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            metrics['precision'] = precision_score(all_targets, all_preds, average='macro', zero_division=0)
            metrics['recall'] = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        return avg_loss, metrics

    def save_checkpoint(self, is_best=False, save_epoch=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint (every epoch)
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save epoch checkpoint (every N epochs)
        if save_epoch:
            epoch_path = self.checkpoint_dir / f'epoch_{self.current_epoch + 1}.pth'
            torch.save(checkpoint, epoch_path)
            print(f"✓ Epoch {self.current_epoch + 1} checkpoint saved")

        # Save best checkpoint (when improvement)
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"✓ Best model saved with val_loss: {self.best_val_loss:.4f}, val_f1: {self.best_val_f1:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.val_metrics = checkpoint.get('val_metrics', [])
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✓ Checkpoint loaded from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('epochs', 100)
        early_stopping_patience = self.config.get('early_stopping_patience', 15)

        print(f"Starting training for {num_epochs} epochs...")
        print(f"Early stopping patience: {early_stopping_patience}")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            if val_metrics:
                print(f"Val Metrics:")
                for metric_name, metric_value in val_metrics.items():
                    print(f"  {metric_name}: {metric_value:.4f}")

            # Check for improvement
            is_best = False
            if val_metrics and val_metrics.get('f1_macro', 0) > self.best_val_f1:
                self.best_val_f1 = val_metrics['f1_macro']
                self.best_val_loss = val_loss
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Save checkpoint
            save_freq = self.config.get('save_freq', 5)
            save_epoch = (epoch + 1) % save_freq == 0
            self.save_checkpoint(is_best=is_best, save_epoch=save_epoch)

            # Call epoch callback (e.g., volume.commit() for Modal)
            if self.epoch_callback is not None:
                try:
                    self.epoch_callback(epoch=epoch + 1)
                except Exception as e:
                    print(f"Warning: Epoch callback failed: {e}")

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n✗ Early stopping triggered after {epoch + 1} epochs")
                print(f"No improvement for {early_stopping_patience} epochs")
                break

            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'val_metrics': self.val_metrics
            }
            with open(self.checkpoint_dir / 'history.json', 'w') as f:
                json.dump(history, f, indent=2)

        print("\n" + "="*60)
        print("Training completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)