import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
import time


class Trainer:
    """Trainer for mouse behavior detection models"""

    def __init__(self, model, train_loader, val_loader, config, device='cuda'):
        """
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Setup optimizer
        self.optimizer = self._build_optimizer()

        # Setup loss function
        self.criterion = self._build_criterion()

        # Setup scheduler
        self.scheduler = self._build_scheduler()

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

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
            criterion = nn.CrossEntropyLoss()
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
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            scheduler = None

        return scheduler

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
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

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute loss
            # Reshape for loss computation: [batch*seq_len, num_classes]
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

            # Compute accuracy
            _, predicted = torch.max(outputs_flat, 1)
            acc = (predicted == targets_flat).float().mean()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0

        return avg_loss, avg_acc

    def validate(self):
        """Validate the model"""
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
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

                # Compute accuracy
                _, predicted = torch.max(outputs_flat, 1)
                acc = (predicted == targets_flat).float().mean()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0

        return avg_loss, avg_acc

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved with val_loss: {self.best_val_loss:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Checkpoint loaded from epoch {self.current_epoch}")

    def train(self):
        """Main training loop"""
        num_epochs = self.config.get('epochs', 100)

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Train
            train_loss, train_acc = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)

            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Print epoch summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(is_best=is_best)

            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
            }
            with open(self.checkpoint_dir / 'history.json', 'w') as f:
                json.dump(history, f)

        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")