"""
Train V8.5 (No Class Weights) on Modal A10G
38 behaviors, Focal Loss only (like V8.1 design)
"""
import modal

app = modal.App("mabe-v8-5-no-weights")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy==2.2.1",
        "pandas==2.2.3",
        "pyarrow==19.0.0",
        "scikit-learn==1.6.1",
        "scipy==1.15.1",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
    )
)


@app.function(
    image=image,
    gpu="A10G",  # 24GB VRAM - sufficient for V8.5
    timeout=3600 * 10,  # 10 hours
    volumes={"/vol": volume},
    memory=32768,  # 32GB RAM
)
def train_v8_5_no_weights(
    config_name: str = "config_v8.5_no_weights.yaml",
):
    """Train V8.5 No Class Weights model on A10G GPU"""
    import torch
    import yaml
    import sys
    from pathlib import Path
    import numpy as np
    from tqdm import tqdm

    print("="*60)
    print("V8.5 No Class Weights Training on Modal A10G")
    print("38 Behaviors, Focal Loss Only (γ=2.0)")
    print("="*60)
    print(f"Config: {config_name}")

    # Add code directory to path
    sys.path.insert(0, "/vol/code")

    # Load config
    config_path = Path("/vol/code/configs") / config_name
    print(f"Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config for Modal/Kaggle data
    config['data_dir'] = '/vol/data/kaggle'
    config['use_kaggle_data'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram_gb:.1f} GB")

    print(f"\nModel: {config['model_type']}")
    print(f"NUM_ACTIONS: {config['num_actions']}")
    print(f"Batch Size: {config.get('batch_size', 256)}")
    print(f"Use Class Weights: {config.get('use_class_weights', False)}")
    print(f"Focal Gamma: {config.get('focal_gamma', 2.0)}")
    print()

    # Import V8.5 modules
    from versions.v8_5_full_behaviors.v8_5_dataset import create_v85_dataloaders
    from versions.v8_5_full_behaviors.v8_5_model import V85BehaviorDetector, V85MultiTaskLoss
    from versions.v8_5_full_behaviors.action_mapping import NUM_ACTIONS, ID_TO_ACTION

    # Validate config
    assert config['num_actions'] == NUM_ACTIONS, f"Config num_actions ({config['num_actions']}) != NUM_ACTIONS ({NUM_ACTIONS})"
    print(f"✓ Config validation passed (num_actions={NUM_ACTIONS})")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_v85_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 256),
        sequence_length=config.get('sequence_length', 100),
        stride=config.get('stride', 25),
        num_workers=0,  # Disable multiprocessing to avoid pin_memory issues
        pin_memory=False,  # Disable pin_memory
    )

    print(f"[OK] Train batches: {len(train_loader)}")
    print(f"[OK] Val batches: {len(val_loader)}")

    # Get input dimension
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"[OK] Input dimension: {input_dim}")

    # Create model
    print("\nCreating V8.5 model...")
    model = V85BehaviorDetector(
        input_dim=input_dim,
        num_actions=NUM_ACTIONS,
        conv_channels=config.get('conv_channels', [128, 256, 512]),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 2),
        dropout=config.get('dropout', 0.0),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Total parameters: {total_params:,}")

    # Loss function (NO class weights)
    criterion = V85MultiTaskLoss(
        action_weight=config.get('action_loss_weight', 1.0),
        agent_weight=config.get('agent_loss_weight', 0.3),
        target_weight=config.get('target_loss_weight', 0.3),
        use_focal=config.get('use_focal_loss', True),
        focal_gamma=config.get('focal_gamma', 2.0),
        class_weights=None  # No class weights!
    )
    print(f"[OK] Loss: Focal (γ={config.get('focal_gamma', 2.0)}), No Class Weights")

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

    # Training function
    def train_epoch():
        model.train()
        total_loss = 0
        action_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc="Training")
        for keypoints, action, agent, target in pbar:
            keypoints = keypoints.to(device)
            action = action.to(device)
            agent = agent.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            action_logits, agent_logits, target_logits = model(keypoints)
            loss, loss_dict = criterion(
                action_logits, agent_logits, target_logits,
                action, agent, target
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
            optimizer.step()

            total_loss += loss.item()
            action_preds = torch.argmax(action_logits, dim=-1)
            action_correct += (action_preds == action).sum().item()
            total_samples += action.numel()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{action_correct/total_samples:.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        acc = action_correct / total_samples
        return avg_loss, acc

    # Validation function
    def validate():
        model.eval()
        total_loss = 0
        action_correct = 0
        total_samples = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for keypoints, action, agent, target in pbar:
                keypoints = keypoints.to(device)
                action = action.to(device)
                agent = agent.to(device)
                target = target.to(device)

                action_logits, agent_logits, target_logits = model(keypoints)
                loss, loss_dict = criterion(
                    action_logits, agent_logits, target_logits,
                    action, agent, target
                )

                total_loss += loss.item()
                action_preds = torch.argmax(action_logits, dim=-1)
                action_correct += (action_preds == action).sum().item()
                total_samples += action.numel()

                all_preds.append(action_preds.cpu().numpy())
                all_labels.append(action.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        acc = action_correct / total_samples

        # Compute per-class metrics
        all_preds = np.concatenate(all_preds).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        return avg_loss, acc, all_preds, all_labels

    # Training loop
    print(f"\n{'='*60}")
    print("Starting Training")
    print(f"{'='*60}\n")

    best_f1 = 0
    best_acc = 0

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch()
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        # Validate
        val_loss, val_acc, val_preds, val_labels = validate()
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # Per-class F1 (sample top behaviors)
        from sklearn.metrics import f1_score
        per_class_f1 = f1_score(val_labels, val_preds, average=None, zero_division=0)

        print(f"\nTop 10 Behaviors (Frame-level F1):")
        top_classes = np.argsort(per_class_f1)[::-1][:10]
        for cls_id in top_classes:
            behavior_name = ID_TO_ACTION.get(cls_id, f'class_{cls_id}')
            print(f"  [{cls_id:2d}] {behavior_name:20s}: F1={per_class_f1[cls_id]:.4f}")

        # Macro F1
        macro_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
        print(f"\nMacro F1: {macro_f1:.4f}")

        # Save best model
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            checkpoint_path = f"/vol/checkpoints/v8_5_no_weights_best_f1.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': macro_f1,
                'acc': val_acc,
            }, checkpoint_path)
            print(f"[SAVE] Best F1 model: {macro_f1:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = f"/vol/checkpoints/v8_5_no_weights_best_acc.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': macro_f1,
                'acc': val_acc,
            }, checkpoint_path)
            print(f"[SAVE] Best Acc model: {val_acc:.4f}")

        scheduler.step()

        # Commit volume every 5 epochs
        if (epoch + 1) % 5 == 0:
            volume.commit()
            print(f"[COMMIT] Volume saved at epoch {epoch+1}")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Best Acc: {best_acc:.4f}")
    print(f"{'='*60}\n")

    volume.commit()
    print("[COMMIT] Final volume save complete")

    return {
        'best_f1': best_f1,
        'best_acc': best_acc,
    }


@app.local_entrypoint()
def main():
    """Entry point for modal run"""
    result = train_v8_5_no_weights.remote()
    print(f"\n✓ Training complete!")
    print(f"  Best F1: {result['best_f1']:.4f}")
    print(f"  Best Acc: {result['best_acc']:.4f}")
