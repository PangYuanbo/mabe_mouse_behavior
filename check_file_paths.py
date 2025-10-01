"""
Check actual file paths in Kaggle data
"""

import modal

app = modal.App("check-paths")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def check_paths():
    import pandas as pd
    from pathlib import Path

    # Read train.csv
    train_csv = pd.read_csv("/vol/data/kaggle/train.csv")
    print(f"Total videos in train.csv: {len(train_csv)}")

    # Get first video
    first_row = train_csv.iloc[0]
    video_id = first_row['video_id']
    lab_id = first_row['lab_id']

    print(f"\nFirst video:")
    print(f"  lab_id: {lab_id}")
    print(f"  video_id: {video_id}")

    # Check if tracking file exists
    tracking_file = Path(f"/vol/data/kaggle/train_tracking/{lab_id}/{video_id}.parquet")
    print(f"\nLooking for tracking file:")
    print(f"  {tracking_file}")
    print(f"  Exists: {tracking_file.exists()}")

    # Check if annotation file exists
    annotation_file = Path(f"/vol/data/kaggle/train_annotation/{lab_id}/{video_id}.parquet")
    print(f"\nLooking for annotation file:")
    print(f"  {annotation_file}")
    print(f"  Exists: {annotation_file.exists()}")

    # List files in train_tracking
    tracking_dir = Path("/vol/data/kaggle/train_tracking")
    if tracking_dir.exists():
        print(f"\nDirectories in {tracking_dir}:")
        for item in sorted(tracking_dir.iterdir())[:5]:
            print(f"  {item.name}")
            if item.is_dir():
                files = list(item.glob("*.parquet"))
                print(f"    Files: {len(files)}")
                if files:
                    print(f"    First file: {files[0].name}")

    # List files in train_annotation
    annotation_dir = Path("/vol/data/kaggle/train_annotation")
    if annotation_dir.exists():
        print(f"\nDirectories in {annotation_dir}:")
        for item in sorted(annotation_dir.iterdir())[:5]:
            print(f"  {item.name}")
            if item.is_dir():
                files = list(item.glob("*.parquet"))
                print(f"    Files: {len(files)}")
                if files:
                    print(f"    First file: {files[0].name}")


@app.local_entrypoint()
def main():
    check_paths.remote()


if __name__ == "__main__":
    main()
