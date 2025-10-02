"""
Download Kaggle competition data directly to Modal volume
"""

import modal

app = modal.App("mabe-download-kaggle")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Image with kaggle installed
image = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=3600,  # 1 hour for download
    secrets=[modal.Secret.from_name("kaggle-secret")],  # Will create this
)
def download_data():
    """Download MABe competition data from Kaggle"""
    import os
    import subprocess
    from pathlib import Path

    print("="*60)
    print("Downloading MABe Competition Data from Kaggle")
    print("="*60)

    # Kaggle credentials should be set as Modal secret
    # The secret should have KAGGLE_USERNAME and KAGGLE_KEY

    # Create data directory
    data_dir = Path("/vol/data/kaggle")
    data_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nDownloading to: {data_dir}")

    # Download competition data
    competition_name = "MABe-mouse-behavior-detection"

    try:
        # List competition files first
        print(f"\nListing files for competition: {competition_name}")
        result = subprocess.run(
            ["kaggle", "competitions", "files", "-c", competition_name],
            capture_output=True,
            text=True
        )
        print(result.stdout)

        # Download all data
        print(f"\nDownloading competition data...")
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", competition_name, "-p", str(data_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)

        # Unzip files
        print("\nUnzipping files...")
        import zipfile

        for zip_file in data_dir.glob("*.zip"):
            print(f"  Extracting: {zip_file.name}")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(data_dir)
            # Remove zip file to save space
            zip_file.unlink()
            print(f"  ✓ Extracted and removed {zip_file.name}")

        # List downloaded files
        print("\n" + "="*60)
        print("Downloaded files:")
        print("="*60)
        for file in sorted(data_dir.rglob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024*1024)
                print(f"  {file.relative_to(data_dir)} ({size_mb:.2f} MB)")

        volume.commit()

        print("\n" + "="*60)
        print("✓ Download complete!")
        print("="*60)

        return str(data_dir)

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error downloading data: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise


@app.local_entrypoint()
def main():
    """Download Kaggle data to Modal"""
    print("\n" + "="*60)
    print("Starting Kaggle data download on Modal...")
    print("="*60)
    print("\nNote: Make sure you have created the 'kaggle-secret' secret")
    print("Run: modal secret create kaggle-secret KAGGLE_USERNAME=<your_username> KAGGLE_KEY=<your_key>")
    print("="*60 + "\n")

    result = download_data.remote()

    print(f"\n✓ Data downloaded to Modal: {result}")


if __name__ == "__main__":
    main()
