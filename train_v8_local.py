"""
V8 Training Script - Multi-task Fine-grained Behavior Detection
Run: python train_v8_local.py --config configs/config_v8_5090.yaml
"""

import torch
import yaml
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from versions.v8_fine_grained.v8_dataset import create_v8_dataloaders, V8Dataset
from versions.v8_fine_grained.v8_model import V8BehaviorDetector, V8MultiTaskLoss
from versions.v8_fine_grained.action_mapping import NUM_ACTIONS, ID_TO_ACTION
from versions.v8_fine_grained.submission_utils import predictions_to_intervals, evaluate_intervals
from versions.v8_fine_grained.advanced_postprocessing import (
    temporal_smoothing,
    probs_to_intervals_advanced
)
from src.utils.detailed_metrics import (
    compute_per_class_metrics,
    compute_interval_per_class_f1,
    print_detailed_metrics
)


def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    total_action_loss = 0
    total_agent_loss = 0
    total_target_loss = 0

    action_correct = 0
    agent_correct = 0
    target_correct = 0
    total_samples = 0

    pbar = tqdm(train_loader, desc="Training")

    for batch_idx, (keypoints, action, agent, target) in enumerate(pbar):
        keypoints = keypoints.to(device)
        action = action.to(device)
        agent = agent.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                action_logits, agent_logits, target_logits = model(keypoints)
                loss, loss_dict = criterion(
                    action_logits, agent_logits, target_logits,
                    action, agent, target
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            action_logits, agent_logits, target_logits = model(keypoints)
            loss, loss_dict = criterion(
                action_logits, agent_logits, target_logits,
                action, agent, target
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Compute accuracies
        action_pred = action_logits.argmax(dim=-1)
        agent_pred = agent_logits.argmax(dim=-1)
        target_pred = target_logits.argmax(dim=-1)

        action_correct += (action_pred == action).sum().item()
        agent_correct += (agent_pred == agent).sum().item()
        target_correct += (target_pred == target).sum().item()

        B, T = action.shape
        total_samples += B * T

        total_loss += loss_dict['total']
        total_action_loss += loss_dict['action']
        total_agent_loss += loss_dict['agent']
        total_target_loss += loss_dict['target']

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'act_acc': f"{action_correct / total_samples:.4f}",
            'agt_acc': f"{agent_correct / total_samples:.4f}"
        })

    avg_loss = total_loss / len(train_loader)
    action_acc = action_correct / total_samples
    agent_acc = agent_correct / total_samples
    target_acc = target_correct / total_samples

    return {
        'loss': avg_loss,
        'action_loss': total_action_loss / len(train_loader),
        'agent_loss': total_agent_loss / len(train_loader),
        'target_loss': total_target_loss / len(train_loader),
        'action_acc': action_acc,
        'agent_acc': agent_acc,
        'target_acc': target_acc
    }


@torch.no_grad()
def validate(model, val_loader, criterion, device, compute_interval_f1=False, use_advanced_postproc=False):
    """Validate model"""
    model.eval()

    total_loss = 0
    total_action_loss = 0
    total_agent_loss = 0
    total_target_loss = 0

    action_correct = 0
    agent_correct = 0
    target_correct = 0
    total_samples = 0

    # For interval F1 computation
    all_action_preds = []
    all_agent_preds = []
    all_target_preds = []
    all_action_labels = []
    all_agent_labels = []
    all_target_labels = []

    # For advanced postprocessing (collect probabilities)
    all_action_probs = []
    all_agent_probs = []
    all_target_probs = []

    for keypoints, action, agent, target in tqdm(val_loader, desc="Validation"):
        keypoints = keypoints.to(device)
        action = action.to(device)
        agent = agent.to(device)
        target = target.to(device)

        action_logits, agent_logits, target_logits = model(keypoints)
        loss, loss_dict = criterion(
            action_logits, agent_logits, target_logits,
            action, agent, target
        )

        # Compute accuracies
        action_pred = action_logits.argmax(dim=-1)
        agent_pred = agent_logits.argmax(dim=-1)
        target_pred = target_logits.argmax(dim=-1)

        action_correct += (action_pred == action).sum().item()
        agent_correct += (agent_pred == agent).sum().item()
        target_correct += (target_pred == target).sum().item()

        B, T = action.shape
        total_samples += B * T

        total_loss += loss_dict['total']
        total_action_loss += loss_dict['action']
        total_agent_loss += loss_dict['agent']
        total_target_loss += loss_dict['target']

        # Collect predictions for interval F1
        if compute_interval_f1:
            all_action_preds.append(action_pred.cpu().numpy())
            all_agent_preds.append(agent_pred.cpu().numpy())
            all_target_preds.append(target_pred.cpu().numpy())
            all_action_labels.append(action.cpu().numpy())
            all_agent_labels.append(agent.cpu().numpy())
            all_target_labels.append(target.cpu().numpy())

            # Collect probabilities for advanced postprocessing
            if use_advanced_postproc:
                all_action_probs.append(torch.softmax(action_logits, dim=-1).cpu().numpy())
                all_agent_probs.append(torch.softmax(agent_logits, dim=-1).cpu().numpy())
                all_target_probs.append(torch.softmax(target_logits, dim=-1).cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    action_acc = action_correct / total_samples
    agent_acc = agent_correct / total_samples
    target_acc = target_correct / total_samples

    metrics = {
        'loss': avg_loss,
        'action_loss': total_action_loss / len(val_loader),
        'agent_loss': total_agent_loss / len(val_loader),
        'target_loss': total_target_loss / len(val_loader),
        'action_acc': action_acc,
        'agent_acc': agent_acc,
        'target_acc': target_acc
    }

    # Compute detailed metrics if requested
    if compute_interval_f1:
        # Concatenate all predictions (flatten batch dimension)
        action_preds_flat = np.concatenate(all_action_preds, axis=0).flatten()
        agent_preds_flat = np.concatenate(all_agent_preds, axis=0).flatten()
        target_preds_flat = np.concatenate(all_target_preds, axis=0).flatten()
        action_labels_flat = np.concatenate(all_action_labels, axis=0).flatten()
        agent_labels_flat = np.concatenate(all_agent_labels, axis=0).flatten()
        target_labels_flat = np.concatenate(all_target_labels, axis=0).flatten()

        # 1. Per-class frame-level metrics
        frame_class_metrics = compute_per_class_metrics(
            action_preds_flat,
            action_labels_flat,
            num_classes=NUM_ACTIONS
        )
        metrics['frame_class_metrics'] = frame_class_metrics

        # 2. Convert to intervals
        if use_advanced_postproc:
            # Advanced postprocessing pipeline
            print("  [Postproc] Using advanced postprocessing...")

            # Concatenate probabilities
            action_probs_concat = np.concatenate(all_action_probs, axis=0)  # [N, T, C]
            agent_probs_concat = np.concatenate(all_agent_probs, axis=0)
            target_probs_concat = np.concatenate(all_target_probs, axis=0)

            # Flatten batch dimension
            action_probs_flat = action_probs_concat.reshape(-1, action_probs_concat.shape[-1])
            agent_probs_flat = agent_probs_concat.reshape(-1, agent_probs_concat.shape[-1])
            target_probs_flat = target_probs_concat.reshape(-1, target_probs_concat.shape[-1])

            # Temporal smoothing
            action_probs_smooth = temporal_smoothing(action_probs_flat, kernel_size=5, method='median')
            agent_probs_smooth = temporal_smoothing(agent_probs_flat, kernel_size=5, method='median')
            target_probs_smooth = temporal_smoothing(target_probs_flat, kernel_size=5, method='median')

            # Convert to intervals with advanced postprocessing
            pred_intervals = probs_to_intervals_advanced(
                action_probs=action_probs_smooth,
                agent_probs=agent_probs_smooth,
                target_probs=target_probs_smooth,
                action_names=ID_TO_ACTION,
                keypoints=None  # No velocity filtering during training validation
            )
        else:
            # Basic postprocessing (argmax + simple filtering)
            pred_intervals = predictions_to_intervals(
                action_preds=action_preds_flat,
                agent_preds=agent_preds_flat,
                target_preds=target_preds_flat,
                min_duration=5
            )

        gt_intervals = predictions_to_intervals(
            action_preds=action_labels_flat,
            agent_preds=agent_labels_flat,
            target_preds=target_labels_flat,
            min_duration=1  # No filtering for ground truth
        )

        # 3. Overall interval F1
        interval_metrics = evaluate_intervals(pred_intervals, gt_intervals, iou_threshold=0.5)
        metrics['interval_f1'] = interval_metrics['f1']
        metrics['interval_precision'] = interval_metrics['precision']
        metrics['interval_recall'] = interval_metrics['recall']

        # 4. Per-class interval F1
        interval_class_metrics = compute_interval_per_class_f1(
            pred_intervals,
            gt_intervals,
            num_classes=NUM_ACTIONS,
            iou_threshold=0.5
        )
        metrics['interval_class_metrics'] = interval_class_metrics

    return metrics


def main():
    parser = argparse.ArgumentParser(description='V8 Multi-task Training')
    parser.add_argument('--config', type=str, default='configs/config_v8_5090.yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Device
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("V8 Multi-task Fine-grained Behavior Detection")
    print("="*60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Create dataloaders
    print(f"\nLoading data...")
    train_loader, val_loader = create_v8_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length'],
        stride=config.get('stride', 25),
        num_workers=config.get('num_workers', 0),
        pin_memory=config.get('pin_memory', True)
    )

    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")

    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"[OK] Input dimension: {input_dim}")

    # Build model
    print(f"\nBuilding V8 model...")
    model = V8BehaviorDetector(
        input_dim=input_dim,
        num_actions=NUM_ACTIONS,
        num_mice=config['num_mice'],
        conv_channels=config['conv_channels'],
        lstm_hidden=config['lstm_hidden'],
        lstm_layers=config['lstm_layers'],
        dropout=config['dropout']
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Total parameters: {total_params:,}")

    # Loss function
    criterion = V8MultiTaskLoss(
        action_weight=config.get('action_loss_weight', 1.0),
        agent_weight=config.get('agent_loss_weight', 0.3),
        target_weight=config.get('target_loss_weight', 0.3),
        use_focal=config.get('use_focal_loss', True),
        focal_gamma=config.get('focal_gamma', 2.0)
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
        eta_min=config.get('min_lr', 1e-5)
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.get('use_amp', True) else None

    # Training loop
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    best_val_acc = 0
    patience_counter = 0

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Checkpoint dir: {checkpoint_dir}\n")

    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler)

        # Validate (compute interval F1 every 5 epochs)
        compute_f1 = (epoch % 5 == 0)
        use_advanced = config.get('use_advanced_postprocessing', False)
        val_metrics = validate(
            model, val_loader, criterion, device,
            compute_interval_f1=compute_f1,
            use_advanced_postproc=use_advanced
        )

        # Print metrics
        print(f"\n[Train] Loss: {train_metrics['loss']:.4f} | "
              f"Action Acc: {train_metrics['action_acc']:.4f} | "
              f"Agent Acc: {train_metrics['agent_acc']:.4f} | "
              f"Target Acc: {train_metrics['target_acc']:.4f}")

        if 'interval_f1' in val_metrics:
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
                  f"Action Acc: {val_metrics['action_acc']:.4f} | "
                  f"Agent Acc: {val_metrics['agent_acc']:.4f} | "
                  f"Target Acc: {val_metrics['target_acc']:.4f}")
            print(f"[Kaggle] Interval F1: {val_metrics['interval_f1']:.4f} | "
                  f"Precision: {val_metrics['interval_precision']:.4f} | "
                  f"Recall: {val_metrics['interval_recall']:.4f}")

            # Print detailed per-class metrics
            if 'frame_class_metrics' in val_metrics and 'interval_class_metrics' in val_metrics:
                print_detailed_metrics(
                    frame_metrics=val_metrics['frame_class_metrics'],
                    interval_metrics=val_metrics['interval_class_metrics'],
                    action_names=ID_TO_ACTION,
                    top_k=15  # Show top 15 classes
                )
        else:
            print(f"[Val]   Loss: {val_metrics['loss']:.4f} | "
                  f"Action Acc: {val_metrics['action_acc']:.4f} | "
                  f"Agent Acc: {val_metrics['agent_acc']:.4f} | "
                  f"Target Acc: {val_metrics['target_acc']:.4f}")

        # Save best model
        if val_metrics['action_acc'] > best_val_acc:
            best_val_acc = val_metrics['action_acc']
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pth')
            print(f"\n[OK] Saved best model (val_acc={best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # Always save latest model
        torch.save(model.state_dict(), checkpoint_dir / 'last_model.pth')
        
        # Early stopping
        if patience_counter >= config.get('early_stopping_patience', 20):
            print(f"\n[X] Early stopping triggered (patience={patience_counter})")
            break

        # Step scheduler
        scheduler.step()

    print(f"\n{'='*60}")
    print(f"[OK] Training complete!")
    print(f"  Best Val Action Acc: {best_val_acc:.4f}")
    print(f"  Checkpoint: {checkpoint_dir / 'best_model.pth'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
