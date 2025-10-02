"""
Quick check of train.csv structure
"""

import modal

app = modal.App("check-csv")
volume = modal.Volume.from_name("mabe-data")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pandas")


@app.function(
    image=image,
    volumes={"/vol": volume},
)
def check_csv():
    import pandas as pd
    from pathlib import Path

    csv_path = Path("/vol/data/kaggle/train.csv")
    df = pd.read_csv(csv_path)

    print("train.csv structure:")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst 10 rows:")
    print(df.head(10))
    print(f"\nData types:")
    print(df.dtypes)

    return list(df.columns)


@app.local_entrypoint()
def main():
    columns = check_csv.remote()
    print(f"\n✓ Columns found: {columns}")


if __name__ == "__main__":
    main()
