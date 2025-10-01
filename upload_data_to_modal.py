"""
Upload local data to Modal volume
"""

import modal

app = modal.App("mabe-upload-data")

# Create/get volume
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)

# Simple image with Python
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=3600,
)
def upload_data(local_files: dict):
    """
    Upload local data files to Modal volume

    Args:
        local_files: Dict mapping local paths to remote paths
    """
    from pathlib import Path
    import base64

    uploaded = []

    for remote_path, content_b64 in local_files.items():
        remote_file = Path("/vol") / remote_path
        remote_file.parent.mkdir(parents=True, exist_ok=True)

        # Decode base64 content
        content = base64.b64decode(content_b64)

        with open(remote_file, "wb") as f:
            f.write(content)

        uploaded.append(remote_path)
        print(f"✓ Uploaded: {remote_path}")

    # Commit changes to volume
    volume.commit()

    return uploaded


@app.local_entrypoint()
def main():
    """
    Upload local data directory to Modal volume
    """
    from pathlib import Path
    import base64

    print("="*60)
    print("Uploading data to Modal volume 'mabe-data'")
    print("="*60)

    local_data_dir = Path("data")

    if not local_data_dir.exists():
        print(f"\nError: Data directory not found: {local_data_dir}")
        print("Please run 'python create_sample_data.py' first")
        return

    # Collect all files
    files_to_upload = {}
    total_size = 0

    print(f"\nScanning {local_data_dir}...")

    for file_path in local_data_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_data_dir)
            remote_path = str(relative_path)

            # Read and encode file
            with open(file_path, "rb") as f:
                content = f.read()
                content_b64 = base64.b64encode(content).decode()
                files_to_upload[remote_path] = content_b64

            file_size = len(content)
            total_size += file_size
            print(f"  Found: {remote_path} ({file_size / 1024:.1f} KB)")

    print(f"\nTotal files: {len(files_to_upload)}")
    print(f"Total size: {total_size / 1024 / 1024:.2f} MB")

    if not files_to_upload:
        print("\nNo files to upload!")
        return

    print("\nUploading to Modal...")

    # Upload in chunks to avoid payload size limits
    chunk_size = 10
    file_items = list(files_to_upload.items())

    for i in range(0, len(file_items), chunk_size):
        chunk = dict(file_items[i:i+chunk_size])
        print(f"\nUploading chunk {i//chunk_size + 1}/{(len(file_items)-1)//chunk_size + 1}...")
        upload_data.remote(chunk)

    print("\n" + "="*60)
    print("✓ Upload complete!")
    print("="*60)
    print("\nData is now available in Modal volume 'mabe-data'")
    print("You can now run: modal run modal_train.py")


if __name__ == "__main__":
    main()