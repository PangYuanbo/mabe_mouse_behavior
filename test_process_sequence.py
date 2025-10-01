"""
Test the _process_sequence method directly
"""

import modal

app = modal.App("test-process-sequence")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas", "pyarrow", "numpy"
)


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=300,
)
def test_process():
    import pandas as pd
    import numpy as np
    from pathlib import Path

    # Test with first video from AdaptableSnail
    lab_id = "AdaptableSnail"
    video_id = "44566106"

    print("="*60)
    print(f"Testing _process_sequence for {lab_id}/{video_id}")
    print("="*60)

    # Load files
    tracking_file = Path(f"/vol/data/kaggle/train_tracking/{lab_id}/{video_id}.parquet")
    annotation_file = Path(f"/vol/data/kaggle/train_annotation/{lab_id}/{video_id}.parquet")

    print(f"\nLoading tracking: {tracking_file}")
    tracking_df = pd.read_parquet(tracking_file)
    print(f"Tracking shape: {tracking_df.shape}")
    print(f"Tracking columns: {list(tracking_df.columns)}")
    print(f"First few rows:")
    print(tracking_df.head())

    print(f"\nLoading annotation: {annotation_file}")
    annotation_df = pd.read_parquet(annotation_file)
    print(f"Annotation shape: {annotation_df.shape}")
    print(f"Annotation columns: {list(annotation_df.columns)}")
    print(f"First few rows:")
    print(annotation_df.head())

    # Test pivot
    print("\n" + "="*60)
    print("Testing pivot operation...")
    print("="*60)
    try:
        tracking_pivot = tracking_df.pivot_table(
            index='video_frame',
            columns=['mouse_id', 'bodypart'],
            values=['x', 'y'],
            aggfunc='first'
        )
        print(f"✓ Pivot successful!")
        print(f"Pivot shape: {tracking_pivot.shape}")
        print(f"First few rows:")
        print(tracking_pivot.head())

        # Flatten columns
        tracking_pivot.columns = ['_'.join(map(str, col)).strip() for col in tracking_pivot.columns.values]
        tracking_pivot = tracking_pivot.sort_index()
        print(f"\n✓ Flattened columns:")
        print(f"Shape: {tracking_pivot.shape}")
        print(f"Columns ({len(tracking_pivot.columns)}):")
        print(list(tracking_pivot.columns))

        keypoints = tracking_pivot.values.astype(np.float32)
        keypoints = np.nan_to_num(keypoints, nan=0.0)
        print(f"\n✓ Keypoints array shape: {keypoints.shape}")

    except Exception as e:
        print(f"✗ Pivot failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test annotation processing
    print("\n" + "="*60)
    print("Testing annotation processing...")
    print("="*60)
    try:
        num_frames = len(tracking_pivot)
        labels = np.zeros(num_frames, dtype=np.int64)

        action_mapping = {
            'investigation': 1,
            'mount': 2,
            'attack': 3,
            'rear': 0,
            'avoid': 0,
            'other': 0,
        }

        print(f"Number of frames: {num_frames}")
        print(f"Number of events: {len(annotation_df)}")

        for _, row in annotation_df.iterrows():
            action = row['action']
            start_frame = row['start_frame']
            stop_frame = row['stop_frame']

            label = action_mapping.get(action, 0)

            for frame in range(start_frame, stop_frame + 1):
                if frame < num_frames:
                    if label > labels[frame]:
                        labels[frame] = label

        print(f"\n✓ Labels array shape: {labels.shape}")
        print(f"Label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Label {u}: {c} frames ({c/len(labels)*100:.1f}%)")

        print("\n" + "="*60)
        print("✓ SUCCESS: Data processing works!")
        print("="*60)
        print(f"Final keypoints shape: {keypoints.shape}")
        print(f"Final labels shape: {labels.shape}")

    except Exception as e:
        print(f"✗ Annotation processing failed: {e}")
        import traceback
        traceback.print_exc()


@app.local_entrypoint()
def main():
    test_process.remote()


if __name__ == "__main__":
    main()
