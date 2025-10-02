"""
Evaluate trained model on validation set
"""

import modal

app = modal.App("evaluate-model")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.8.0",
    "numpy==2.3.3",
    "pandas==2.3.3",
    "pyarrow==21.0.0",
    "scikit-learn==1.7.2",
    "scipy==1.16.2",
    "pyyaml==6.0.2",
    "tqdm==4.67.1",
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={"/vol": volume},
)
def evaluate_model(
    checkpoint_path: str = "/vol/checkpoints/kaggle/best_model.pth",
    max_sequences: int = 20,
):
    """Evaluate model and generate detailed performance report"""
    import torch
    import yaml
    import sys
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from collections import Counter

    print("="*60)
    print("Model Evaluation on Kaggle Data")
    print("="*60)

    # Add code directory to path
    sys.path.insert(0, "/vol/code")

    # Load config
    config_path = Path("/vol/code/configs/config_advanced.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config
    config['data_dir'] = '/vol/data/kaggle'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Import modules
    from src.data.kaggle_dataset import create_kaggle_dataloaders
    from src.models.advanced_models import Conv1DBiLSTM

    # Load validation data
    print("\nLoading validation data...")
    _, val_loader = create_kaggle_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config.get('batch_size', 32),
        sequence_length=config.get('sequence_length', 100),
        num_workers=2,  # Reduced to avoid crashes
        use_feature_engineering=False,
        max_sequences=max_sequences,
    )

    print(f"✓ Validation batches: {len(val_loader)}")

    # Get input dimension
    sample_batch = next(iter(val_loader))
    input_dim = sample_batch[0].shape[-1]
    print(f"✓ Input dimension: {input_dim}")

    # Build model
    print("\nBuilding model...")
    model = Conv1DBiLSTM(
        input_dim=input_dim,
        num_classes=config['num_classes'],
        conv_channels=config.get('conv_channels', [64, 128, 256]),
        lstm_hidden=config.get('lstm_hidden', 256),
        lstm_layers=config.get('lstm_layers', 2),
        dropout=config.get('dropout', 0.3),
    )

    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    if not Path(checkpoint_path).exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    if 'val_loss' in checkpoint:
        print(f"✓ Checkpoint val_loss: {checkpoint['val_loss']:.4f}")
    if 'val_f1' in checkpoint:
        print(f"✓ Checkpoint val_f1: {checkpoint['val_f1']:.4f}")

    print("\nCheckpoint keys:", list(checkpoint.keys()))

    # Evaluate
    print("\n" + "="*60)
    print("Running evaluation...")
    print("="*60)

    all_predictions = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)  # [batch, seq_len, num_classes]

            # Flatten predictions and labels
            outputs_flat = outputs.reshape(-1, outputs.size(-1))  # [batch*seq_len, num_classes]
            labels_flat = labels.reshape(-1)  # [batch*seq_len]

            # Get predictions
            _, predictions = torch.max(outputs_flat, 1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels_flat.cpu().numpy())
            all_logits.append(outputs_flat.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_logits = np.vstack(all_logits)

    # Calculate metrics
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)

    # Overall metrics
    accuracy = np.mean(all_predictions == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Macro:  {f1:.4f}")

    # Per-class metrics
    print("\n" + "="*60)
    print("PER-CLASS METRICS")
    print("="*60)

    class_names = ['Background', 'Social Investigation', 'Mating', 'Aggressive']

    # Check which classes are present
    unique_labels = np.unique(np.concatenate([all_labels, all_predictions]))
    print(f"\nClasses present in data: {unique_labels}")

    # Only use class names for classes that exist
    labels_param = list(range(4))
    target_names_filtered = [class_names[i] for i in labels_param]

    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=labels_param,
        target_names=target_names_filtered,
        zero_division=0
    ))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n       Predicted:")
    print(f"         {' '.join([f'{i:>8}' for i in range(4)])}")
    print("Actual:")
    for i, row in enumerate(cm):
        print(f"  {i}  {class_names[i]:20s}  {' '.join([f'{val:>8}' for val in row])}")

    # Label distribution
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION")
    print("="*60)

    label_counts = Counter(all_labels)
    pred_counts = Counter(all_predictions)
    total = len(all_labels)

    print("\nActual Labels:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / total * 100
        print(f"  {label} ({class_names[label]:20s}): {count:>8,} ({pct:5.2f}%)")

    print("\nPredicted Labels:")
    for label in sorted(pred_counts.keys()):
        count = pred_counts[label]
        pct = count / total * 100
        print(f"  {label} ({class_names[label]:20s}): {count:>8,} ({pct:5.2f}%)")

    # Model confidence analysis
    print("\n" + "="*60)
    print("MODEL CONFIDENCE ANALYSIS")
    print("="*60)

    probabilities = torch.softmax(torch.from_numpy(all_logits), dim=1).numpy()
    max_probs = np.max(probabilities, axis=1)

    print(f"\nConfidence Statistics:")
    print(f"  Mean confidence: {np.mean(max_probs):.4f}")
    print(f"  Median confidence: {np.median(max_probs):.4f}")
    print(f"  Min confidence: {np.min(max_probs):.4f}")
    print(f"  Max confidence: {np.max(max_probs):.4f}")

    # Confidence by correctness
    correct_mask = all_predictions == all_labels
    print(f"\nConfidence for correct predictions: {np.mean(max_probs[correct_mask]):.4f}")
    print(f"Confidence for incorrect predictions: {np.mean(max_probs[~correct_mask]):.4f}")

    # Behavior detection performance
    print("\n" + "="*60)
    print("BEHAVIOR DETECTION PERFORMANCE")
    print("="*60)

    # Binary: behavior (1,2,3) vs background (0)
    binary_labels = (all_labels > 0).astype(int)
    binary_preds = (all_predictions > 0).astype(int)

    binary_acc = np.mean(binary_labels == binary_preds)
    binary_f1 = f1_score(binary_labels, binary_preds)

    print(f"\nBehavior vs Background (Binary):")
    print(f"  Accuracy: {binary_acc:.4f}")
    print(f"  F1 Score: {binary_f1:.4f}")

    # Per behavior type
    for behavior_id in [1, 2, 3]:
        behavior_mask = all_labels == behavior_id
        if behavior_mask.sum() > 0:
            behavior_recall = np.mean(all_predictions[behavior_mask] == behavior_id)
            behavior_precision = (
                np.sum((all_predictions == behavior_id) & (all_labels == behavior_id)) /
                max(np.sum(all_predictions == behavior_id), 1)
            )
            print(f"\n{class_names[behavior_id]}:")
            print(f"  Recall (捕获率):    {behavior_recall:.4f}")
            print(f"  Precision (准确率): {behavior_precision:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n✓ Total frames evaluated: {total:,}")
    print(f"✓ Overall accuracy: {accuracy:.4f}")
    print(f"✓ F1 Macro: {f1:.4f}")
    print(f"✓ Behavior detection F1: {binary_f1:.4f}")

    return {
        'accuracy': accuracy,
        'f1_macro': f1,
        'precision': precision,
        'recall': recall,
        'binary_f1': binary_f1,
        'confusion_matrix': cm.tolist(),
    }


@app.local_entrypoint()
def main(
    checkpoint: str = "best_model.pth",
    max_sequences: int = 20,
):
    """
    Evaluate model on validation set

    Args:
        checkpoint: Checkpoint filename
        max_sequences: Max sequences to evaluate (for quick testing)
    """
    checkpoint_path = f"/vol/checkpoints/kaggle/{checkpoint}"

    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Max sequences: {max_sequences}")
    print("="*60 + "\n")

    result = evaluate_model.remote(
        checkpoint_path=checkpoint_path,
        max_sequences=max_sequences,
    )

    if result:
        print("\n" + "="*60)
        print("✓ Evaluation complete!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ Evaluation failed")
        print("="*60)


if __name__ == "__main__":
    main()
