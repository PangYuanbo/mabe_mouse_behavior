"""
Modal Jupyter Notebook for MABe Mouse Behavior Detection
Interactive environment for data exploration, training, and submission preparation
"""

import modal

# Create Modal app
app = modal.App("mabe-notebook")

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.0.0",
        "kaggle>=1.7.0",
    )
)

# Volume for data and checkpoints
volume = modal.Volume.from_name("mabe-data", create_if_missing=True)


@app.function(
    image=image,
    gpu="T4",
    timeout=28800,  # 8 hours
    volumes={"/vol": volume},
)
@modal.web_server(8888, startup_timeout=60)
def run_jupyter():
    """Launch Jupyter Lab server on Modal"""
    import subprocess
    import os

    # Set Jupyter configuration
    os.environ["JUPYTER_ENABLE_LAB"] = "yes"

    # Start Jupyter Lab
    subprocess.Popen([
        "jupyter", "lab",
        "--ip=0.0.0.0",
        "--port=8888",
        "--no-browser",
        "--allow-root",
        "--NotebookApp.token=''",
        "--NotebookApp.password=''",
        "--NotebookApp.allow_origin='*'",
        "--NotebookApp.base_url=/",
    ])


# Alternative: Create a notebook execution function
@app.function(
    image=image,
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={"/vol": volume},
)
def run_notebook(notebook_path: str):
    """
    Execute a Jupyter notebook on Modal

    Args:
        notebook_path: Path to notebook file in volume

    Returns:
        Executed notebook content
    """
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from pathlib import Path

    # Read notebook
    notebook_file = Path(f"/vol/{notebook_path}")

    with open(notebook_file, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    # Execute notebook
    ep = ExecutePreprocessor(timeout=14400, kernel_name='python3')

    try:
        ep.preprocess(nb, {'metadata': {'path': '/vol/'}})
        print(f"✓ Notebook executed successfully: {notebook_path}")
    except Exception as e:
        print(f"✗ Error executing notebook: {e}")
        raise

    # Save executed notebook
    output_path = notebook_file.with_stem(f"{notebook_file.stem}_executed")

    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"✓ Executed notebook saved to: {output_path}")

    return str(output_path)


# Create a starter notebook
@app.local_entrypoint()
def create_starter_notebook():
    """Create a starter notebook for MABe analysis"""

    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# MABe Mouse Behavior Detection - Starter Notebook\n\n",
                          "This notebook provides a starting point for exploring the MABe dataset, ",
                          "training models, and preparing submissions."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import torch\n",
                    "import yaml\n",
                    "from pathlib import Path\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_style('whitegrid')\n",
                    "%matplotlib inline"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Load Data"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Data paths\n",
                    "data_dir = Path('/vol')\n",
                    "train_dir = data_dir / 'train'\n",
                    "val_dir = data_dir / 'val'\n",
                    "\n",
                    "# List data files\n",
                    "train_files = sorted(list(train_dir.glob('*.npy')))\n",
                    "print(f'Training files: {len(train_files)}')\n",
                    "\n",
                    "# Load a sample\n",
                    "sample_data = np.load(train_files[0])\n",
                    "print(f'Sample shape: {sample_data.shape}')\n",
                    "print(f'Features per frame: {sample_data.shape[1]}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 2. Visualize Keypoints"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Reshape to keypoints\n",
                    "num_mice = 2\n",
                    "num_keypoints = 7\n",
                    "keypoints = sample_data.reshape(-1, num_mice, num_keypoints, 2)\n",
                    "\n",
                    "# Plot first frame\n",
                    "frame_idx = 0\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "\n",
                    "for mouse_idx in range(num_mice):\n",
                    "    kpts = keypoints[frame_idx, mouse_idx]\n",
                    "    plt.scatter(kpts[:, 0], kpts[:, 1], label=f'Mouse {mouse_idx}', s=100)\n",
                    "    \n",
                    "    # Connect keypoints\n",
                    "    plt.plot(kpts[:, 0], kpts[:, 1], alpha=0.5)\n",
                    "\n",
                    "plt.legend()\n",
                    "plt.title(f'Frame {frame_idx} - Mouse Keypoints')\n",
                    "plt.xlabel('X coordinate')\n",
                    "plt.ylabel('Y coordinate')\n",
                    "plt.axis('equal')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 3. Feature Engineering"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "import sys\n",
                    "sys.path.insert(0, '/vol/code')\n",
                    "\n",
                    "from src.data.feature_engineering import MouseFeatureEngineer\n",
                    "\n",
                    "# Create feature engineer\n",
                    "feature_engineer = MouseFeatureEngineer(num_mice=2, num_keypoints=7)\n",
                    "\n",
                    "# Extract features\n",
                    "features = feature_engineer.extract_all_features(\n",
                    "    sample_data,\n",
                    "    include_pca=False,\n",
                    "    include_temporal=True\n",
                    ")\n",
                    "\n",
                    "print(f'Extracted features shape: {features.shape}')\n",
                    "print(f'Number of features: {features.shape[1]}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 4. Load Model"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "from src.models.advanced_models import build_advanced_model\n",
                    "\n",
                    "# Load config\n",
                    "with open('/vol/code/configs/config_advanced.yaml', 'r') as f:\n",
                    "    config = yaml.safe_load(f)\n",
                    "\n",
                    "config['input_dim'] = features.shape[1]\n",
                    "\n",
                    "# Build model\n",
                    "model = build_advanced_model(config)\n",
                    "print(f'Model: {config[\"model_type\"]}')\n",
                    "print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 5. Training"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "from src.utils.trainer import Trainer\n",
                    "from src.data.dataset import get_dataloaders\n",
                    "\n",
                    "# Update paths\n",
                    "config['train_data_dir'] = '/vol/train'\n",
                    "config['val_data_dir'] = '/vol/val'\n",
                    "config['train_annotation_file'] = '/vol/train_annotations.json'\n",
                    "config['val_annotation_file'] = '/vol/val_annotations.json'\n",
                    "config['checkpoint_dir'] = '/vol/checkpoints'\n",
                    "\n",
                    "# Create dataloaders\n",
                    "train_loader, val_loader = get_dataloaders(config)\n",
                    "\n",
                    "# Create trainer\n",
                    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                    "trainer = Trainer(model, train_loader, val_loader, config, device)\n",
                    "\n",
                    "# Train\n",
                    "trainer.train()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 6. Generate Submission"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Load best model\n",
                    "checkpoint = torch.load('/vol/checkpoints/best_model.pth')\n",
                    "model.load_state_dict(checkpoint['model_state_dict'])\n",
                    "model.eval()\n",
                    "\n",
                    "# Generate predictions on test set\n",
                    "test_dir = Path('/vol/test')\n",
                    "test_files = sorted(list(test_dir.glob('*.npy')))\n",
                    "\n",
                    "submissions = []\n",
                    "\n",
                    "for test_file in test_files:\n",
                    "    # Load test data\n",
                    "    test_data = np.load(test_file)\n",
                    "    \n",
                    "    # Extract features\n",
                    "    test_features = feature_engineer.extract_all_features(\n",
                    "        test_data, include_pca=False, include_temporal=True\n",
                    "    )\n",
                    "    \n",
                    "    # Convert to tensor\n",
                    "    test_tensor = torch.FloatTensor(test_features).unsqueeze(0).to(device)\n",
                    "    \n",
                    "    # Predict\n",
                    "    with torch.no_grad():\n",
                    "        logits = model(test_tensor)\n",
                    "        predictions = torch.argmax(logits, dim=-1)\n",
                    "    \n",
                    "    # Store results\n",
                    "    submissions.append({\n",
                    "        'file': test_file.name,\n",
                    "        'predictions': predictions.cpu().numpy()\n",
                    "    })\n",
                    "\n",
                    "print(f'Generated predictions for {len(submissions)} files')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [
                    "# Save submission\n",
                    "import json\n",
                    "\n",
                    "submission_data = {}\n",
                    "for item in submissions:\n",
                    "    submission_data[item['file']] = item['predictions'].tolist()\n",
                    "\n",
                    "with open('/vol/submission.json', 'w') as f:\n",
                    "    json.dump(submission_data, f)\n",
                    "\n",
                    "print('✓ Submission saved to /vol/submission.json')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Save notebook
    import nbformat
    from pathlib import Path

    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(exist_ok=True)

    notebook_path = notebook_dir / "mabe_starter.ipynb"

    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nbformat.from_dict(notebook_content), f)

    print(f"✓ Starter notebook created: {notebook_path}")
    print("\nTo use the notebook on Modal:")
    print("1. Upload notebook: modal run upload_code_to_modal.py")
    print("2. Start Jupyter server: modal run modal_notebook.py::run_jupyter")
    print("3. Or execute notebook: modal run modal_notebook.py::run_notebook --notebook-path notebooks/mabe_starter.ipynb")


if __name__ == "__main__":
    create_starter_notebook()