"""
Inspect column structure of parquet files
"""

import modal

app = modal.App("inspect-columns")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas", "pyarrow")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def inspect_columns():
    import pandas as pd
    from pathlib import Path

    # Read train.csv to get first video
    train_csv = pd.read_csv("/vol/data/kaggle/train.csv")
    first_row = train_csv.iloc[0]
    video_id = first_row['video_id']
    lab_id = first_row['lab_id']

    print("="*60)
    print(f"Inspecting: {lab_id}/{video_id}")
    print("="*60)

    # Load tracking file
    tracking_file = Path(f"/vol/data/kaggle/train_tracking/{lab_id}/{video_id}.parquet")
    print(f"\n📦 Tracking file: {tracking_file}")
    print(f"Exists: {tracking_file.exists()}")

    if tracking_file.exists():
        df = pd.read_parquet(tracking_file)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")

        print(f"\nFirst 5 rows:")
        print(df.head())

        # Check for columns starting with x_ or y_
        x_cols = [col for col in df.columns if col.startswith('x_')]
        y_cols = [col for col in df.columns if col.startswith('y_')]
        print(f"\nColumns starting with 'x_': {len(x_cols)}")
        print(f"Columns starting with 'y_': {len(y_cols)}")

    # Load annotation file
    annotation_file = Path(f"/vol/data/kaggle/train_annotation/{lab_id}/{video_id}.parquet")
    print(f"\n\n📦 Annotation file: {annotation_file}")
    print(f"Exists: {annotation_file.exists()}")

    if annotation_file.exists():
        df = pd.read_parquet(annotation_file)
        print(f"\nShape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for i, col in enumerate(df.columns):
            print(f"  {i}: {col}")

        print(f"\nFirst 10 rows:")
        print(df.head(10))

        print(f"\nData types:")
        print(df.dtypes)


@app.local_entrypoint()
def main():
    inspect_columns.remote()


if __name__ == "__main__":
    main()
