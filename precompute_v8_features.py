"""
Pre-compute V8 features for V9 training
Runs V8 inference on entire dataset and saves outputs to disk
"""

import torch
import numpy as np
import yaml
import argparse
import h5py
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))

from versions.v8_fine_grained.v8_dataset import V8Dataset
from versions.v8_fine_grained.v8_model import V8BehaviorDetector


def precompute_split(
    v8_model,
    dataset,
    output_file,
    device,
    batch_size=256
):
    """
    Precompute V8 features for a dataset split

    Args:
        v8_model: Trained V8 model
        dataset: V8Dataset instance
        output_file: Path to save HDF5 file
        device: Device for inference
        batch_size: Batch size for inference
    """
    v8_model.eval()

    num_sequences = len(dataset)

    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        print(f"\nCreating HDF5 file: {output_file}")
        print(f"Total sequences: {num_sequences}")

        # Pre-allocate datasets (more efficient than appending)
        # We'll store V8 logits for each sequence
        # Shape: [num_sequences, sequence_length, output_dim]

        # Get dimensions from first sample
        # V8Dataset returns tuple: (keypoints, action, agent, target)
        keypoints_sample, _, _, _ = dataset[0]
        seq_length = keypoints_sample.shape[0]

        # Create datasets
        action_logits_dset = f.create_dataset(
            'action_logits',
            shape=(num_sequences, seq_length, 28),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        agent_logits_dset = f.create_dataset(
            'agent_logits',
            shape=(num_sequences, seq_length, 4),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        target_logits_dset = f.create_dataset(
            'target_logits',
            shape=(num_sequences, seq_length, 4),
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        # Also save ground truth labels for V9 training
        action_labels_dset = f.create_dataset(
            'action_labels',
            shape=(num_sequences, seq_length),
            dtype=np.int64,
            compression='gzip',
            compression_opts=4
        )

        agent_labels_dset = f.create_dataset(
            'agent_labels',
            shape=(num_sequences, seq_length),
            dtype=np.int64,
            compression='gzip',
            compression_opts=4
        )

        target_labels_dset = f.create_dataset(
            'target_labels',
            shape=(num_sequences, seq_length),
            dtype=np.int64,
            compression='gzip',
            compression_opts=4
        )

        # Process in batches
        with torch.no_grad():
            for start_idx in tqdm(range(0, num_sequences, batch_size), desc="Precomputing V8 features"):
                end_idx = min(start_idx + batch_size, num_sequences)
                batch_indices = range(start_idx, end_idx)

                # Load batch
                batch_keypoints = []
                batch_actions = []
                batch_agents = []
                batch_targets = []

                for idx in batch_indices:
                    # V8Dataset returns tuple: (keypoints, action, agent, target)
                    keypoints_t, action_t, agent_t, target_t = dataset[idx]
                    batch_keypoints.append(keypoints_t)
                    batch_actions.append(action_t)
                    batch_agents.append(agent_t)
                    batch_targets.append(target_t)

                # Stack into batch
                keypoints = torch.stack(batch_keypoints).to(device)

                # V8 inference
                action_logits, agent_logits, target_logits = v8_model(keypoints)

                # Save to HDF5
                action_logits_dset[start_idx:end_idx] = action_logits.cpu().numpy()
                agent_logits_dset[start_idx:end_idx] = agent_logits.cpu().numpy()
                target_logits_dset[start_idx:end_idx] = target_logits.cpu().numpy()

                # Save labels
                action_labels_dset[start_idx:end_idx] = torch.stack(batch_actions).numpy()
                agent_labels_dset[start_idx:end_idx] = torch.stack(batch_agents).numpy()
                target_labels_dset[start_idx:end_idx] = torch.stack(batch_targets).numpy()

        # Add metadata
        f.attrs['num_sequences'] = num_sequences
        f.attrs['sequence_length'] = seq_length
        f.attrs['action_dim'] = 28
        f.attrs['agent_dim'] = 4
        f.attrs['target_dim'] = 4

    print(f"[OK] Saved to {output_file}")

    # Print file size
    file_size_mb = Path(output_file).stat().st_size / 1024 / 1024
    print(f"  File size: {file_size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Precompute V8 features for V9 training')
    parser.add_argument('--v8_config', type=str, default='configs/config_v8_5090.yaml')
    parser.add_argument('--v8_checkpoint', type=str, default='checkpoints/v8_5090/best_model.pth')
    parser.add_argument('--output_dir', type=str, default='data/v8_features')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    # Load V8 config
    with open(args.v8_config) as f:
        v8_config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*60)
    print("V8 Feature Precomputation for V9 Training")
    print("="*60)
    print(f"Device: {device}")
    print(f"V8 checkpoint: {args.v8_checkpoint}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load V8 model
    print("\nLoading V8 model...")
    v8_model = V8BehaviorDetector(
        input_dim=v8_config['input_dim'],
        num_actions=28,
        num_mice=v8_config['num_mice'],
        conv_channels=v8_config['conv_channels'],
        lstm_hidden=v8_config['lstm_hidden'],
        lstm_layers=v8_config['lstm_layers'],
        dropout=0.0  # Always 0 for inference
    ).to(device)

    v8_model.load_state_dict(torch.load(args.v8_checkpoint, map_location=device))
    v8_model.eval()
    print("[OK] V8 model loaded")

    # Process train split
    print("\n" + "="*60)
    print("Processing TRAIN split")
    print("="*60)

    train_dataset = V8Dataset(
        data_dir=v8_config['data_dir'],
        split='train',
        sequence_length=v8_config['sequence_length'],
        stride=50  # Use stride=50 for V9 (faster)
    )

    precompute_split(
        v8_model=v8_model,
        dataset=train_dataset,
        output_file=output_dir / 'train_v8_features.h5',
        device=device,
        batch_size=args.batch_size
    )

    # Process val split
    print("\n" + "="*60)
    print("Processing VAL split")
    print("="*60)

    val_dataset = V8Dataset(
        data_dir=v8_config['data_dir'],
        split='val',
        sequence_length=v8_config['sequence_length'],
        stride=100  # No overlap for val
    )

    precompute_split(
        v8_model=v8_model,
        dataset=val_dataset,
        output_file=output_dir / 'val_v8_features.h5',
        device=device,
        batch_size=args.batch_size
    )

    print("\n" + "="*60)
    print("[OK] Feature precomputation complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  {output_dir / 'train_v8_features.h5'}")
    print(f"  {output_dir / 'val_v8_features.h5'}")
    print(f"\nNow you can train V9 with precomputed features using:")
    print(f"  python train_v9_precomputed.py --config configs/config_v9_assembler.yaml")


if __name__ == '__main__':
    main()
