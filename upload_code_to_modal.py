"""
Upload source code to Modal volume
"""

import modal

app = modal.App("mabe-upload-code")
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    volumes={"/vol": volume},
    timeout=600,
)
def upload_code(files: dict):
    """Upload source code files to Modal volume"""
    from pathlib import Path
    import base64

    uploaded = []

    for remote_path, content_b64 in files.items():
        remote_file = Path("/vol/code") / remote_path
        remote_file.parent.mkdir(parents=True, exist_ok=True)

        content = base64.b64decode(content_b64)

        with open(remote_file, "wb") as f:
            f.write(content)

        uploaded.append(remote_path)
        print(f"✓ Uploaded: {remote_path}")

    volume.commit()
    return uploaded


@app.local_entrypoint()
def main():
    """Upload source code to Modal"""
    from pathlib import Path
    import base64

    print("="*60)
    print("Uploading source code to Modal volume 'mabe-data'")
    print("="*60)

    # Collect source files
    files_to_upload = {}

    # Upload src directory
    src_dir = Path("src")
    for file_path in src_dir.rglob("*.py"):
        relative_path = str(file_path.relative_to("."  ))
        with open(file_path, "rb") as f:
            content = base64.b64encode(f.read()).decode()
            files_to_upload[relative_path] = content
        print(f"  Found: {relative_path}")

    # Upload all configs
    configs_dir = Path("configs")
    for config_file in configs_dir.glob("*.yaml"):
        relative_path = str(config_file.relative_to("."))
        with open(config_file, "rb") as f:
            content = base64.b64encode(f.read()).decode()
            files_to_upload[relative_path] = content
        print(f"  Found: {relative_path}")

    print(f"\nTotal files: {len(files_to_upload)}")
    print("\nUploading to Modal...")

    upload_code.remote(files_to_upload)

    print("\n" + "="*60)
    print("✓ Code upload complete!")
    print("="*60)


if __name__ == "__main__":
    main()