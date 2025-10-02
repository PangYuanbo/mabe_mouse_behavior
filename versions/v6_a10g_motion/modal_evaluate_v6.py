"""
Evaluate V6 Checkpoint on Modal A10G
Test latest checkpoint performance without stopping training
"""
import modal

app = modal.App("mabe-v6-evaluate")
volume = modal.Volume.from_name("mabe-data")

# Same image as training
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
    timeout=3600,
    volumes={"/vol": volume},
    memory=32768,
)
def evaluate_checkpoint(
    checkpoint_name: str = "latest_checkpoint.pth",
    checkpoint_dir: str = "/vol/checkpoints/v6_a10g",
    max_val_sequences: int = None,  # Limit validation sequences for quick test
):
    """Evaluate V6 checkpoint"""
    import torch
    import yaml
    import sys
    from pathlib import Path
    import numpy as np
    from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

    print("="*60)
    print("MABe V6 Checkpoint Evaluation")
    print("="*60)

    # Add code to path
    sys.path.insert(0, "/vol/code")

    # Load config
    config_path = Path("/vol/code/configs/config_v6_a10g.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config['data_dir'] = '/vol/data/kaggle'
    config['use_kaggle_data'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load checkpoint
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name
    print(f"Loading checkpoint: {checkpoint_path}")

    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found!")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"✓ Checkpoint loaded")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'unknown')}")
    print(f"  Val F1: {checkpoint.get('val_f1', 'unknown')}")
    print()

    # Import modules
    from src.data.kaggle_dataset import KaggleMABeDataset
    from src.models.advanced_models import Conv1DBiLSTM
    from torch.utils.data import DataLoader

    # Create ONLY validation dataset (not training)
    print("Loading validation data...")
    if max_val_sequences:
        print(f"⚠️  Quick test mode: Using only {max_val_sequences} validation videos")
    else:
        print(f"✓ Full validation mode: Using ALL validation videos")

    val_dataset = KaggleMABeDataset(
        data_dir=config['data_dir'],
        split='val',  # Only validation data
        sequence_length=config.get('sequence_length', 100),
        stride=config.get('sequence_length', 100),  # No overlap for validation
        max_sequences=max_val_sequences,  # Limit if specified
        use_feature_engineering=False,
        feature_engineer=None,
    )

    # Force validation dataset to use same input dimension as training (284)
    # This ensures compatibility with the trained model
    val_dataset.max_input_dim = 284  # Same as training
    print(f"✓ Forcing input dimension to 284 (same as training)")

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 384),
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"✓ Loaded {len(val_dataset)} validation sequences")
    print(f"✓ Validation batches: {len(val_loader)}")

    # Get input dimension
    sample_batch = next(iter(val_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"✓ Input dimension: {input_dim}")
    print()

    # Build model
    print("Building model...")
    model = Conv1DBiLSTM(
        input_dim=input_dim,
        num_classes=config['num_classes'],
        conv_channels=config.get('conv_channels', [64, 128, 256]),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 2),
        dropout=config.get('dropout', 0.3),
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✓ Model loaded and ready")
    print()

    # Evaluate
    print("="*60)
    print("Evaluating on validation set...")
    print("="*60)

    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0

    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(config.get('class_weights', [1.0, 5.0, 8.0, 8.0])).to(device)
    )

    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(val_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(sequences)

            # Flatten outputs and labels for loss computation
            outputs_flat = outputs.reshape(-1, outputs.size(-1))  # [batch*seq, num_classes]
            labels_flat = labels.reshape(-1)  # [batch*seq]

            loss = criterion(outputs_flat, labels_flat)

            # Predictions
            _, preds = torch.max(outputs_flat, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    avg_loss = total_loss / num_batches

    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Average Loss: {avg_loss:.4f}")
    print()

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    print(f"Accuracy:         {accuracy:.4f}")
    print(f"F1 Macro:         {f1_macro:.4f}")
    print(f"F1 Weighted:      {f1_weighted:.4f}")
    print(f"Precision Macro:  {precision:.4f}")
    print(f"Recall Macro:     {recall:.4f}")
    print()

    # Per-class metrics
    print("Per-Class Performance:")
    print("-" * 60)
    class_names = ['Background', 'Social', 'Mating', 'Aggressive']

    for i in range(4):
        mask = all_labels == i
        if mask.sum() > 0:
            # Accuracy for this class
            class_acc = (all_preds[mask] == i).mean()

            # F1 for this class (one-vs-rest)
            binary_labels = (all_labels == i).astype(int)
            binary_preds = (all_preds == i).astype(int)
            class_f1 = f1_score(binary_labels, binary_preds, zero_division=0)

            support = mask.sum()
            print(f"{class_names[i]:12s}: Acc={class_acc:.4f}, F1={class_f1:.4f}, Support={support:,}")

    print()
    print("Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    ))

    # Class distribution
    print("Prediction Distribution:")
    print("-" * 60)
    unique, counts = np.unique(all_preds, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = count / len(all_preds) * 100
        print(f"{class_names[cls]:12s}: {count:,} ({pct:.2f}%)")

    print()
    print("Ground Truth Distribution:")
    print("-" * 60)
    unique, counts = np.unique(all_labels, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = count / len(all_labels) * 100
        print(f"{class_names[cls]:12s}: {count:,} ({pct:.2f}%)")

    print()
    print("="*60)
    print("✓ Evaluation complete!")
    print("="*60)

    return {
        'checkpoint': checkpoint_name,
        'epoch': checkpoint.get('epoch', 'unknown'),
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
    }


@app.local_entrypoint()
def main(
    checkpoint: str = "latest_checkpoint.pth",
    max_val: int = None,  # Limit validation sequences (None = all)
):
    """
    Evaluate V6 checkpoint

    Args:
        checkpoint: Checkpoint filename (default: latest_checkpoint.pth)
                   Options: latest_checkpoint.pth, best_model.pth, epoch_N.pth
        max_val: Max validation sequences to use (None = all, 50 = quick test)
    """
    print(f"\nEvaluating checkpoint: {checkpoint}")
    if max_val:
        print(f"Quick test mode: {max_val} validation sequences\n")
    else:
        print(f"Full validation mode: All validation data\n")

    result = evaluate_checkpoint.remote(
        checkpoint_name=checkpoint,
        max_val_sequences=max_val,
    )

    if result:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Checkpoint:  {result['checkpoint']}")
        print(f"Epoch:       {result['epoch']}")
        print(f"Loss:        {result['loss']:.4f}")
        print(f"Accuracy:    {result['accuracy']:.4f}")
        print(f"F1 Macro:    {result['f1_macro']:.4f}")
        print(f"Precision:   {result['precision']:.4f}")
        print(f"Recall:      {result['recall']:.4f}")
        print("="*60)


if __name__ == "__main__":
    main()
