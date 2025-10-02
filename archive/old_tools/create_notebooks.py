"""
Create starter notebooks for MABe analysis
"""

import nbformat as nbf


def create_starter_notebook():
    """Create a starter notebook for MABe analysis"""

    nb = nbf.v4.new_notebook()

    cells = [
        nbf.v4.new_markdown_cell("# MABe Mouse Behavior Detection - Starter Notebook\n\n"
                                 "This notebook provides a starting point for exploring the MABe dataset, "
                                 "training models, and preparing submissions."),

        nbf.v4.new_code_cell(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import torch\n"
            "import yaml\n"
            "from pathlib import Path\n"
            "\n"
            "# Set style\n"
            "sns.set_style('whitegrid')\n"
            "%matplotlib inline"
        ),

        nbf.v4.new_markdown_cell("## 1. Load Data"),

        nbf.v4.new_code_cell(
            "# Data paths\n"
            "data_dir = Path('/vol' if Path('/vol').exists() else 'data')\n"
            "train_dir = data_dir / 'train'\n"
            "val_dir = data_dir / 'val'\n"
            "\n"
            "# List data files\n"
            "train_files = sorted(list(train_dir.glob('*.npy')))\n"
            "print(f'Training files: {len(train_files)}')\n"
            "\n"
            "# Load a sample\n"
            "sample_data = np.load(train_files[0])\n"
            "print(f'Sample shape: {sample_data.shape}')\n"
            "print(f'Features per frame: {sample_data.shape[1]}')"
        ),

        nbf.v4.new_markdown_cell("## 2. Visualize Keypoints"),

        nbf.v4.new_code_cell(
            "# Reshape to keypoints\n"
            "num_mice = 2\n"
            "num_keypoints = 7\n"
            "keypoints = sample_data.reshape(-1, num_mice, num_keypoints, 2)\n"
            "\n"
            "# Plot first frame\n"
            "frame_idx = 0\n"
            "plt.figure(figsize=(10, 8))\n"
            "\n"
            "for mouse_idx in range(num_mice):\n"
            "    kpts = keypoints[frame_idx, mouse_idx]\n"
            "    plt.scatter(kpts[:, 0], kpts[:, 1], label=f'Mouse {mouse_idx}', s=100)\n"
            "    \n"
            "    # Connect keypoints\n"
            "    plt.plot(kpts[:, 0], kpts[:, 1], alpha=0.5)\n"
            "\n"
            "plt.legend()\n"
            "plt.title(f'Frame {frame_idx} - Mouse Keypoints')\n"
            "plt.xlabel('X coordinate')\n"
            "plt.ylabel('Y coordinate')\n"
            "plt.axis('equal')\n"
            "plt.show()"
        ),

        nbf.v4.new_markdown_cell("## 3. Analyze Temporal Dynamics"),

        nbf.v4.new_code_cell(
            "# Plot inter-mouse distance over time\n"
            "mouse0_centroid = keypoints[:, 0, :, :].mean(axis=1)\n"
            "mouse1_centroid = keypoints[:, 1, :, :].mean(axis=1)\n"
            "inter_dist = np.linalg.norm(mouse0_centroid - mouse1_centroid, axis=1)\n"
            "\n"
            "plt.figure(figsize=(12, 4))\n"
            "plt.plot(inter_dist)\n"
            "plt.title('Inter-Mouse Distance Over Time')\n"
            "plt.xlabel('Frame')\n"
            "plt.ylabel('Distance')\n"
            "plt.grid(True)\n"
            "plt.show()\n"
            "\n"
            "print(f'Mean distance: {inter_dist.mean():.3f}')\n"
            "print(f'Std distance: {inter_dist.std():.3f}')"
        ),

        nbf.v4.new_markdown_cell("## 4. Feature Engineering"),

        nbf.v4.new_code_cell(
            "import sys\n"
            "sys.path.insert(0, '/vol/code' if Path('/vol/code').exists() else 'src')\n"
            "\n"
            "from data.feature_engineering import MouseFeatureEngineer\n"
            "\n"
            "# Create feature engineer\n"
            "feature_engineer = MouseFeatureEngineer(num_mice=2, num_keypoints=7)\n"
            "\n"
            "# Extract features\n"
            "features = feature_engineer.extract_all_features(\n"
            "    sample_data,\n"
            "    include_pca=False,\n"
            "    include_temporal=True\n"
            ")\n"
            "\n"
            "print(f'Extracted features shape: {features.shape}')\n"
            "print(f'Number of features: {features.shape[1]}')"
        ),

        nbf.v4.new_markdown_cell("## 5. Load and Test Model"),

        nbf.v4.new_code_cell(
            "from models.advanced_models import build_advanced_model\n"
            "\n"
            "# Load config\n"
            "config_path = '/vol/code/configs/config_advanced.yaml' if Path('/vol/code').exists() else 'configs/config_advanced.yaml'\n"
            "with open(config_path, 'r') as f:\n"
            "    config = yaml.safe_load(f)\n"
            "\n"
            "config['input_dim'] = features.shape[1]\n"
            "\n"
            "# Build model\n"
            "model = build_advanced_model(config)\n"
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
            "model = model.to(device)\n"
            "\n"
            "print(f'Model: {config[\"model_type\"]}')\n"
            "print(f'Device: {device}')\n"
            "print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')"
        ),

        nbf.v4.new_markdown_cell("## 6. Test Inference"),

        nbf.v4.new_code_cell(
            "# Prepare test input\n"
            "test_sequence = torch.FloatTensor(features[:100]).unsqueeze(0).to(device)\n"
            "print(f'Test sequence shape: {test_sequence.shape}')\n"
            "\n"
            "# Run inference\n"
            "model.eval()\n"
            "with torch.no_grad():\n"
            "    logits = model(test_sequence)\n"
            "    predictions = torch.argmax(logits, dim=-1)\n"
            "\n"
            "print(f'Output shape: {logits.shape}')\n"
            "print(f'Predictions shape: {predictions.shape}')\n"
            "print(f'Sample predictions: {predictions[0, :10].cpu().numpy()}')"
        ),

        nbf.v4.new_markdown_cell("## 7. Visualize Predictions"),

        nbf.v4.new_code_cell(
            "# Behavior classes\n"
            "behavior_classes = {0: 'other', 1: 'close_investigation', 2: 'mount', 3: 'attack'}\n"
            "\n"
            "# Plot predictions\n"
            "pred_np = predictions[0].cpu().numpy()\n"
            "\n"
            "plt.figure(figsize=(14, 4))\n"
            "plt.plot(pred_np, marker='o', markersize=3, linestyle='-', alpha=0.7)\n"
            "plt.title('Predicted Behavior Over Time')\n"
            "plt.xlabel('Frame')\n"
            "plt.ylabel('Behavior Class')\n"
            "plt.yticks(range(4), [behavior_classes[i] for i in range(4)])\n"
            "plt.grid(True, alpha=0.3)\n"
            "plt.show()\n"
            "\n"
            "# Count behavior occurrences\n"
            "unique, counts = np.unique(pred_np, return_counts=True)\n"
            "for behavior_id, count in zip(unique, counts):\n"
            "    print(f'{behavior_classes[behavior_id]}: {count} frames ({count/len(pred_np)*100:.1f}%)')"
        ),
    ]

    nb['cells'] = cells

    return nb


def main():
    """Create and save notebooks"""

    from pathlib import Path

    # Create notebooks directory
    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(exist_ok=True)

    # Create starter notebook
    print("Creating starter notebook...")
    starter_nb = create_starter_notebook()

    notebook_path = notebook_dir / "mabe_starter.ipynb"
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(starter_nb, f)

    print(f"✓ Starter notebook created: {notebook_path}")
    print("\nTo use the notebook:")
    print("1. Local: jupyter lab notebooks/mabe_starter.ipynb")
    print("2. Modal: Upload code and run modal_notebook.py")


if __name__ == "__main__":
    main()