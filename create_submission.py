"""
Create Kaggle submission file using best trained model on Modal
Generates predictions for test set and formats for submission
"""

import modal

app = modal.App("mabe-submission")
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
    gpu="A10G",  # GPU for inference
    timeout=3600 * 2,  # 2 hours
    volumes={"/vol": volume},
    memory=32768,  # 32GB RAM
)
def generate_submission(
    checkpoint_name: str = "best_model.pth",
    config_name: str = "config_advanced.yaml",
):
    """Generate Kaggle submission file"""
    import torch
    import yaml
    import sys
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tqdm import tqdm

    print("="*60)
    print("MABe Kaggle Submission Generator")
    print("="*60)

    # Add code directory to path
    sys.path.insert(0, "/vol/code")

    # Load config
    config_path = Path("/vol/code/configs") / config_name
    print(f"Loading config from {config_path}...")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Update config for Kaggle data
    config['data_dir'] = '/vol/data/kaggle'
    config['use_kaggle_data'] = True

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    print()

    # Import modules
    from src.data.kaggle_dataset import KaggleMABeDataset
    from src.models.advanced_models import Conv1DBiLSTM, TemporalConvNet, HybridModel

    # Load checkpoint
    checkpoint_path = Path(f"/vol/checkpoints/h100/{checkpoint_name}")
    if not checkpoint_path.exists():
        # Try A10G checkpoint path
        checkpoint_path = Path(f"/vol/checkpoints/{checkpoint_name}")

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print(f"✓ Checkpoint loaded from Epoch {checkpoint['epoch'] + 1}")
    print(f"  Best Val Loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Best Val F1: {checkpoint['best_val_f1']:.4f}")
    print()

    # Get input dimension from checkpoint
    input_dim = checkpoint['config'].get('input_dim', 288)
    print(f"Input dimension: {input_dim}")

    # Build model
    model_type = checkpoint['config'].get('model_type', 'conv_bilstm')

    if model_type == 'conv_bilstm':
        model = Conv1DBiLSTM(
            input_dim=input_dim,
            num_classes=checkpoint['config']['num_classes'],
            conv_channels=checkpoint['config'].get('conv_channels', [64, 128, 256]),
            lstm_hidden=checkpoint['config'].get('lstm_hidden', 256),
            lstm_layers=checkpoint['config'].get('lstm_layers', 2),
            dropout=checkpoint['config'].get('dropout', 0.3),
        )
    elif model_type == 'tcn':
        model = TemporalConvNet(
            input_dim=input_dim,
            num_classes=checkpoint['config']['num_classes'],
            num_channels=checkpoint['config'].get('tcn_channels', [64, 128, 256, 256]),
            kernel_size=checkpoint['config'].get('tcn_kernel_size', 3),
            dropout=checkpoint['config'].get('dropout', 0.3),
        )
    elif model_type == 'hybrid':
        model = HybridModel(
            input_dim=input_dim,
            num_classes=checkpoint['config']['num_classes'],
            num_keypoints=checkpoint['config'].get('num_keypoints', 18),
            pointnet_dim=checkpoint['config'].get('pointnet_dim', 128),
            temporal_model=checkpoint['config'].get('temporal_model', 'lstm'),
            lstm_hidden=checkpoint['config'].get('lstm_hidden', 256),
            lstm_layers=checkpoint['config'].get('lstm_layers', 2),
            dropout=checkpoint['config'].get('dropout', 0.3),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded: {model_type}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Load test data
    print("Loading test dataset...")
    test_dataset = KaggleMABeDataset(
        data_dir=config['data_dir'],
        split='test',
        sequence_length=config.get('sequence_length', 100),
        use_feature_engineering=False,
        max_sequences=None,  # Use all test data
    )

    print(f"✓ Test sequences: {len(test_dataset)}")
    print()

    # Generate predictions
    print("Generating predictions...")
    all_predictions = []

    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset)), desc="Predicting"):
            sequence, label, metadata = test_dataset[idx]

            # Add batch dimension
            sequence = sequence.unsqueeze(0).to(device)

            # Forward pass
            output = model(sequence)
            probs = torch.softmax(output, dim=-1)

            # Get predictions for each frame
            probs_np = probs.squeeze(0).cpu().numpy()  # [T, num_classes]
            preds = np.argmax(probs_np, axis=1)  # [T]

            # Extract metadata
            video_id = metadata['video_id']
            start_frame = metadata['start_frame']

            # Create submission rows for each frame
            for i, (pred, prob) in enumerate(zip(preds, probs_np)):
                frame_id = start_frame + i
                all_predictions.append({
                    'video_id': video_id,
                    'frame': frame_id,
                    'prediction': int(pred),
                    'prob_0': prob[0],  # background
                    'prob_1': prob[1],  # social
                    'prob_2': prob[2],  # mating
                    'prob_3': prob[3],  # aggressive
                })

    # Create submission DataFrame
    submission_df = pd.DataFrame(all_predictions)

    # Sort by video_id and frame
    submission_df = submission_df.sort_values(['video_id', 'frame']).reset_index(drop=True)

    print()
    print("="*60)
    print("Submission Summary")
    print("="*60)
    print(f"Total predictions: {len(submission_df):,}")
    print(f"Unique videos: {submission_df['video_id'].nunique()}")
    print()
    print("Prediction distribution:")
    pred_counts = submission_df['prediction'].value_counts().sort_index()
    class_names = {0: 'Background', 1: 'Social', 2: 'Mating', 3: 'Aggressive'}
    for class_id, count in pred_counts.items():
        pct = count / len(submission_df) * 100
        print(f"  {class_names[class_id]:12s}: {count:8,} ({pct:5.2f}%)")
    print("="*60)
    print()

    # Save submission file
    submission_path = Path("/vol/submissions/submission.csv")
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with and without probabilities
    submission_df[['video_id', 'frame', 'prediction']].to_csv(
        submission_path,
        index=False
    )

    submission_df.to_csv(
        Path("/vol/submissions/submission_with_probs.csv"),
        index=False
    )

    # Commit volume
    print("⏳ Committing volume...")
    volume.commit()
    print("✓ Volume committed")

    print(f"✓ Submission saved to: {submission_path}")
    print(f"✓ Full submission saved to: /vol/submissions/submission_with_probs.csv")
    print()

    return {
        'total_predictions': len(submission_df),
        'unique_videos': int(submission_df['video_id'].nunique()),
        'prediction_distribution': pred_counts.to_dict(),
    }


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def download_submission():
    """Download submission file"""
    from pathlib import Path

    submission_path = Path("/vol/submissions/submission.csv")

    if not submission_path.exists():
        print(f"✗ Submission file not found: {submission_path}")
        return None

    with open(submission_path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(
    checkpoint: str = "best_model.pth",
    config: str = "config_advanced.yaml",
    download: bool = False,
):
    """
    Generate Kaggle submission

    Args:
        checkpoint: Checkpoint file name (default: best_model.pth)
        config: Config file name (default: config_advanced.yaml)
        download: Download submission file to local (default: False)
    """
    print("\n" + "="*60)
    print("MABe Kaggle Submission Generator")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Config: {config}")
    print("="*60 + "\n")

    if download:
        print("Downloading submission file...")
        content = download_submission.remote()
        if content:
            with open("submission.csv", "wb") as f:
                f.write(content)
            print("✓ Downloaded to: submission.csv")
        else:
            print("✗ Download failed")
    else:
        # Generate submission
        result = generate_submission.remote(
            checkpoint_name=checkpoint,
            config_name=config,
        )

        print("\n" + "="*60)
        print("✓ Submission generation completed!")
        print("="*60)
        print(f"Total predictions: {result['total_predictions']:,}")
        print(f"Unique videos: {result['unique_videos']}")
        print("\nTo download submission file:")
        print("  modal run create_submission.py --download")
        print("="*60)


if __name__ == "__main__":
    main()
