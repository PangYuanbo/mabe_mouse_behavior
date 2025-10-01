"""
Inspect Kaggle MABe data structure
"""

import modal

app = modal.App("mabe-inspect-data")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "pandas", "pyarrow", "numpy"
)


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def inspect_data():
    """Inspect the downloaded Kaggle data"""
    import pandas as pd
    from pathlib import Path
    import numpy as np

    print("="*60)
    print("Inspecting MABe Kaggle Data")
    print("="*60)

    # Check directory structure
    kaggle_dir = Path("/vol/data/kaggle")

    print("\n📂 Directory structure:")
    for item in sorted(kaggle_dir.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / (1024*1024)
            rel_path = item.relative_to(kaggle_dir)
            if size_mb > 0.1:  # Only show files > 0.1 MB
                print(f"  {rel_path} ({size_mb:.2f} MB)")

    # Read metadata CSVs
    print("\n" + "="*60)
    print("📄 train.csv")
    print("="*60)
    train_csv = pd.read_csv(kaggle_dir / "train.csv")
    print(f"Shape: {train_csv.shape}")
    print(f"Columns: {list(train_csv.columns)}")
    print("\nFirst few rows:")
    print(train_csv.head())

    print("\n" + "="*60)
    print("📄 test.csv")
    print("="*60)
    test_csv = pd.read_csv(kaggle_dir / "test.csv")
    print(f"Shape: {test_csv.shape}")
    print(f"Columns: {list(test_csv.columns)}")
    print("\nFirst few rows:")
    print(test_csv.head())

    # Inspect a sample parquet file
    print("\n" + "="*60)
    print("📦 Sample tracking data (parquet)")
    print("="*60)

    # Find first train annotation file
    annotation_files = list((kaggle_dir / "train_annotation").rglob("*.parquet"))
    if annotation_files:
        sample_file = annotation_files[0]
        print(f"File: {sample_file.relative_to(kaggle_dir)}")
        df = pd.read_parquet(sample_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head(10))
        print("\nData types:")
        print(df.dtypes)
        print("\nValue counts for annotation columns:")
        for col in df.columns:
            if col not in ['sequence_id', 'frame']:
                print(f"\n{col}:")
                print(df[col].value_counts())

    # Check tracking data
    print("\n" + "="*60)
    print("📦 Sample test tracking data")
    print("="*60)

    tracking_files = list((kaggle_dir / "test_tracking").rglob("*.parquet"))
    if tracking_files:
        sample_file = tracking_files[0]
        print(f"File: {sample_file.relative_to(kaggle_dir)}")
        df = pd.read_parquet(sample_file)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head(10))
        print("\nColumn data types:")
        print(df.dtypes)

    # Count datasets and sequences
    print("\n" + "="*60)
    print("📊 Dataset Statistics")
    print("="*60)

    datasets = [d.name for d in (kaggle_dir / "train_annotation").iterdir() if d.is_dir()]
    print(f"\nNumber of datasets: {len(datasets)}")
    print(f"Dataset names: {datasets}")

    for dataset in datasets[:3]:  # Check first 3 datasets
        anno_files = list((kaggle_dir / "train_annotation" / dataset).glob("*.parquet"))
        print(f"\n{dataset}: {len(anno_files)} annotation files")

    print("\n" + "="*60)


@app.local_entrypoint()
def main():
    inspect_data.remote()


if __name__ == "__main__":
    main()
