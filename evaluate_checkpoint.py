"""
Evaluate a specific checkpoint on validation data
"""

import modal

app = modal.App("mabe-checkpoint-eval")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.8.0",
        "numpy==2.3.3",
        "pandas==2.3.3",
        "pyarrow==21.0.0",
        "scikit-learn==1.7.2",
        "scipy==1.16.2",
        "pyyaml==6.0.2",
        "tqdm==4.67.1",
    )
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,  # 1 hour
    volumes={"/vol": volume},
    memory=16384,  # 16GB RAM
)
def evaluate_checkpoint(
    checkpoint_name: str = "epoch_5.pth",
    config_name: str = "config_advanced.yaml",
):
    """Evaluate a checkpoint on validation data"""
    import torch
    import yaml
    import sys
    from pathlib import Path
    import numpy as np
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
    from tqdm import tqdm

    print("="*60)
    print(f"Evaluating Checkpoint: {checkpoint_name}")
    print("="*60)

    # Add code directory to path
    sys.path.insert(0, "/vol/code")

    # Load config
    config_path = Path("/vol/code/configs") / config_name
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config for Kaggle data
    config['data_dir'] = '/vol/data/kaggle'
    config['use_kaggle_data'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Import modules
    from src.data.kaggle_dataset import create_kaggle_dataloaders
    from src.models.advanced_models import Conv1DBiLSTM, TemporalConvNet, HybridModel

    # Create validation dataloader only
    print("\nLoading validation data...")
    _, val_loader = create_kaggle_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 64),
        sequence_length=config.get('sequence_length', 100),
        num_workers=2,
        use_feature_engineering=False,
        max_sequences=None,
    )

    print(f"✓ Validation batches: {len(val_loader)}")

    # Get input dimension from first batch
    sample_batch = next(iter(val_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"✓ Input dimension: {input_dim}")

    # Build model
    print("\nBuilding model...")
    model_type = config.get('model_type', 'conv_bilstm')

    if model_type == 'conv_bilstm':
        model = Conv1DBiLSTM(
            input_dim=input_dim,
            num_classes=config['num_classes'],
            conv_channels=config.get('conv_channels', [64, 128, 256]),
            lstm_hidden=config.get('lstm_hidden', 256),
            lstm_layers=config.get('lstm_layers', 2),
            dropout=config.get('dropout', 0.3),
        )
    elif model_type == 'tcn':
        model = TemporalConvNet(
            input_dim=input_dim,
            num_classes=config['num_classes'],
            num_channels=config.get('tcn_channels', [64, 128, 256, 256]),
            kernel_size=config.get('tcn_kernel_size', 3),
            dropout=config.get('dropout', 0.3),
        )
    elif model_type == 'hybrid':
        model = HybridModel(
            input_dim=input_dim,
            num_classes=config['num_classes'],
            num_keypoints=config.get('num_keypoints', 7),
            pointnet_dim=config.get('pointnet_dim', 128),
            temporal_model=config.get('temporal_model', 'lstm'),
            lstm_hidden=config.get('lstm_hidden', 256),
            lstm_layers=config.get('lstm_layers', 2),
            dropout=config.get('dropout', 0.3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    print(f"✓ Model: {model_type}")

    # Load checkpoint
    checkpoint_path = Path("/vol/checkpoints/kaggle") / checkpoint_name
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Checkpoint from Epoch {checkpoint['epoch'] + 1}")
    print(f"✓ Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"✓ Best Val F1: {checkpoint.get('best_val_f1', 'N/A')}")

    # Evaluate
    print("\n" + "="*60)
    print("Starting Evaluation...")
    print("="*60)

    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0
    num_batches = 0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            if len(batch) == 2:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
            else:
                continue

            # Flatten input if needed
            batch_size, seq_len, *input_shape = inputs.shape
            if len(input_shape) > 1:
                inputs = inputs.reshape(batch_size, seq_len, -1)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            outputs_flat = outputs.reshape(-1, outputs.size(-1))
            targets_flat = targets.reshape(-1)
            loss = criterion(outputs_flat, targets_flat)

            # Collect predictions
            _, predicted = torch.max(outputs_flat, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets_flat.cpu().numpy())

            total_loss += loss.item()
            num_batches += 1

    # Compute metrics
    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    print(f"\nAverage Loss: {avg_loss:.4f}")

    if len(all_preds) > 0:
        accuracy = accuracy_score(all_targets, all_preds)
        f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

        print(f"\nOverall Metrics:")
        print(f"  Accuracy:     {accuracy:.4f}")
        print(f"  F1 (Macro):   {f1_macro:.4f}")
        print(f"  F1 (Weighted): {f1_weighted:.4f}")
        print(f"  Precision:    {precision:.4f}")
        print(f"  Recall:       {recall:.4f}")

        # Per-class metrics
        print(f"\nPer-Class Metrics:")
        class_names = ['Background', 'Social Investigation', 'Mating', 'Aggressive']

        # Per-class F1, Precision, Recall
        f1_per_class = f1_score(all_targets, all_preds, average=None, zero_division=0)
        precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_preds, average=None, zero_division=0)

        for i, class_name in enumerate(class_names):
            if i < len(f1_per_class):
                print(f"\n  {class_name} (Class {i}):")
                print(f"    F1:        {f1_per_class[i]:.4f}")
                print(f"    Precision: {precision_per_class[i]:.4f}")
                print(f"    Recall:    {recall_per_class[i]:.4f}")

        # Full classification report
        print("\n" + "="*60)
        print("Detailed Classification Report:")
        print("="*60)
        print(classification_report(
            all_targets,
            all_preds,
            target_names=class_names,
            zero_division=0
        ))

    print("="*60)
    print("✓ Evaluation Complete!")
    print("="*60)

    return {
        'checkpoint': checkpoint_name,
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
    }


@app.local_entrypoint()
def main(
    checkpoint: str = "epoch_5.pth",
    config: str = "config_advanced.yaml",
):
    """
    Evaluate a specific checkpoint

    Args:
        checkpoint: Checkpoint filename (e.g., 'epoch_5.pth', 'best_model.pth')
        config: Config file name
    """
    print("\n" + "="*60)
    print(f"MABe Checkpoint Evaluation")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {config}")
    print("="*60 + "\n")

    # Evaluate
    result = evaluate_checkpoint.remote(
        checkpoint_name=checkpoint,
        config_name=config,
    )

    print("\n" + "="*60)
    print("✓ Evaluation Summary")
    print("="*60)
    print(f"Checkpoint:    {result['checkpoint']}")
    print(f"Loss:          {result['loss']:.4f}")
    print(f"Accuracy:      {result['accuracy']:.4f}")
    print(f"F1 (Macro):    {result['f1_macro']:.4f}")
    print(f"F1 (Weighted): {result['f1_weighted']:.4f}")
    print(f"Precision:     {result['precision']:.4f}")
    print(f"Recall:        {result['recall']:.4f}")
    print("="*60)


if __name__ == "__main__":
    main()
