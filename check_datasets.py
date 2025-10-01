"""
Check which datasets have annotation files
"""

import modal

app = modal.App("check-datasets")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def check_datasets():
    import pandas as pd
    from pathlib import Path

    # Read train.csv
    train_csv = pd.read_csv("/vol/data/kaggle/train.csv")
    print(f"Total videos in train.csv: {len(train_csv)}")
    print(f"\nLab IDs and counts:")
    print(train_csv['lab_id'].value_counts())

    # Check annotation directory
    annotation_dir = Path("/vol/data/kaggle/train_annotation")
    print(f"\n\nDatasets with annotation files:")
    for lab_dir in sorted(annotation_dir.iterdir()):
        if lab_dir.is_dir():
            files = list(lab_dir.glob("*.parquet"))
            print(f"  {lab_dir.name}: {len(files)} annotation files")

    # Check how many videos from each lab have annotations
    print(f"\n\nChecking which videos have annotations:")
    for lab_id in train_csv['lab_id'].unique():
        videos_in_csv = train_csv[train_csv['lab_id'] == lab_id]['video_id'].values
        annotation_files = list((annotation_dir / lab_id).glob("*.parquet")) if (annotation_dir / lab_id).exists() else []
        annotation_ids = [f.stem for f in annotation_files]

        have_annotations = sum(1 for vid in videos_in_csv if str(vid) in annotation_ids)
        print(f"  {lab_id}: {have_annotations}/{len(videos_in_csv)} videos have annotations")


@app.local_entrypoint()
def main():
    check_datasets.remote()


if __name__ == "__main__":
    main()
