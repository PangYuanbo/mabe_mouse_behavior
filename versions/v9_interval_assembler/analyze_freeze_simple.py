"""
Simple analysis of freeze vs background classes using direct V8 dataset access.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from v8_fine_grained.v8_dataset import V8Dataset
from v8_fine_grained.action_mapping import ACTION_TO_ID


def compute_velocity_simple(keypoints_seq):
    """
    Compute velocity from keypoints sequence.
    Input: [T, 112] (56 coords + 28 speeds + 28 accels)
    Output: average velocity in px/s
    """
    # Extract just the coordinates (first 56 features)
    coords = keypoints_seq[:, :56]  # [T, 56]
    T = coords.shape[0]
    
    if T < 2:
        return 0.0
    
    # Reshape to [T, 28, 2] (28 keypoints, 2 coordinates)
    coords_reshaped = coords.reshape(T, 28, 2)
    
    # Compute frame-to-frame differences
    diffs = np.diff(coords_reshaped, axis=0)  # [T-1, 28, 2]
    
    # Compute speeds (assuming 30 fps)
    speeds = np.linalg.norm(diffs, axis=2) * 30  # [T-1, 28]
    
    # Average across all keypoints and frames
    avg_velocity = np.mean(speeds)
    
    return avg_velocity


def compute_mouse_velocity(keypoints_seq, mouse_idx):
    """
    Compute velocity for a specific mouse.
    Input: [T, 112]
    Output: average velocity in px/s for this mouse
    """
    # Extract coordinates for this mouse (7 bodyparts * 2 coords = 14 features per mouse)
    coords = keypoints_seq[:, :56]  # [T, 56]
    T = coords.shape[0]
    
    if T < 2:
        return 0.0
    
    # Each mouse has 7 bodyparts, 2 coords each = 14 features
    # Mice are arranged sequentially: mouse0 [0:14], mouse1 [14:28], etc.
    start_idx = mouse_idx * 14
    end_idx = start_idx + 14
    
    mouse_coords = coords[:, start_idx:end_idx]  # [T, 14]
    mouse_coords_reshaped = mouse_coords.reshape(T, 7, 2)  # [T, 7 bodyparts, 2]
    
    # Compute frame-to-frame differences
    diffs = np.diff(mouse_coords_reshaped, axis=0)  # [T-1, 7, 2]
    
    # Compute speeds
    speeds = np.linalg.norm(diffs, axis=2) * 30  # [T-1, 7]
    
    # Average across bodyparts and frames
    avg_velocity = np.mean(speeds)
    
    return avg_velocity


def analyze_dataset(dataset, class_name, action_id, max_samples=500):
    """
    Analyze characteristics of a specific class.
    
    Returns:
        Dictionary with statistics
    """
    print(f"\nAnalyzing class: {class_name} (ID: {action_id})")
    
    samples = []
    count = 0
    
    # Go through all sequences
    for idx in range(len(dataset)):
        if count >= max_samples:
            break
        
        try:
            keypoints, action, agent, target = dataset[idx]
            
            # Convert to numpy
            keypoints = keypoints.numpy()  # [T, 112]
            action = action.numpy()        # [T]
            agent = agent.numpy()          # [T]
            target = target.numpy()        # [T]
            
            # Find frames with this action
            if action_id == 0:
                # Background is where action == 0
                matching_frames = (action == 0)
            else:
                matching_frames = (action == action_id)
            
            if not matching_frames.any():
                continue
            
            # Extract segments
            indices = np.where(matching_frames)[0]
            
            if len(indices) == 0:
                continue
            
            # For each frame with this action, compute features
            for frame_idx in indices:
                if count >= max_samples:
                    break
                
                # Get agent and target for this frame
                agent_idx = agent[frame_idx]
                target_idx = target[frame_idx]
                
                # Compute velocities
                # We'll use a small window around this frame
                window_start = max(0, frame_idx - 5)
                window_end = min(len(keypoints), frame_idx + 6)
                
                window_keypoints = keypoints[window_start:window_end]
                
                # Compute velocities
                agent_vel = compute_mouse_velocity(window_keypoints, agent_idx)
                target_vel = compute_mouse_velocity(window_keypoints, target_idx)
                overall_vel = compute_velocity_simple(window_keypoints)
                
                # Compute inter-mouse distance
                coords = window_keypoints[:, :56]
                coords_reshaped = coords.reshape(len(coords), 28, 2)
                
                # Get centroids for agent and target
                agent_start = agent_idx * 7
                agent_end = agent_start + 7
                target_start = target_idx * 7
                target_end = target_start + 7
                
                agent_centroids = np.mean(coords_reshaped[:, agent_start:agent_end, :], axis=1)
                target_centroids = np.mean(coords_reshaped[:, target_start:target_end, :], axis=1)
                
                distances = np.linalg.norm(agent_centroids - target_centroids, axis=1)
                avg_distance = np.mean(distances)
                
                samples.append({
                    'agent_velocity': agent_vel,
                    'target_velocity': target_vel,
                    'overall_velocity': overall_vel,
                    'inter_mouse_distance': avg_distance,
                })
                
                count += 1
        
        except Exception as e:
            print(f"Error processing sequence {idx}: {e}")
            continue
    
    print(f"Collected {count} samples for {class_name}")
    
    if count == 0:
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(samples)
    
    return df


def compare_and_visualize(df_freeze, df_background, output_dir):
    """
    Compare and visualize differences between freeze and background.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)
    
    features = ['agent_velocity', 'target_velocity', 'overall_velocity', 'inter_mouse_distance']
    
    comparison = []
    for feature in features:
        freeze_mean = df_freeze[feature].mean()
        freeze_std = df_freeze[feature].std()
        freeze_median = df_freeze[feature].median()
        
        bg_mean = df_background[feature].mean()
        bg_std = df_background[feature].std()
        bg_median = df_background[feature].median()
        
        # Cohen's d effect size
        pooled_std = np.sqrt((freeze_std**2 + bg_std**2) / 2)
        cohens_d = (freeze_mean - bg_mean) / pooled_std if pooled_std > 0 else 0
        
        comparison.append({
            'feature': feature,
            'freeze_mean': freeze_mean,
            'freeze_std': freeze_std,
            'freeze_median': freeze_median,
            'background_mean': bg_mean,
            'background_std': bg_std,
            'background_median': bg_median,
            'difference': freeze_mean - bg_mean,
            'cohens_d': cohens_d
        })
        
        print(f"\n{feature}:")
        print(f"  Freeze:     mean={freeze_mean:7.2f}, std={freeze_std:6.2f}, median={freeze_median:7.2f}")
        print(f"  Background: mean={bg_mean:7.2f}, std={bg_std:6.2f}, median={bg_median:7.2f}")
        print(f"  Difference: {freeze_mean - bg_mean:7.2f} (Cohen's d={cohens_d:.3f})")
    
    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(output_dir / 'comparison.csv', index=False)
    print(f"\nSaved comparison to: {output_dir / 'comparison.csv'}")
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        freeze_data = df_freeze[feature].dropna()
        bg_data = df_background[feature].dropna()
        
        ax.hist(freeze_data, bins=50, alpha=0.6, label='Freeze', density=True, color='blue')
        ax.hist(bg_data, bins=50, alpha=0.6, label='Background', density=True, color='orange')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature}\n(Freeze: {freeze_data.mean():.2f}, BG: {bg_data.mean():.2f})')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'distributions.png', dpi=150)
    print(f"Saved: {output_dir / 'distributions.png'}")
    plt.close()
    
    # 2. Box plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        data_to_plot = [df_freeze[feature].dropna(), df_background[feature].dropna()]
        ax.boxplot(data_to_plot, labels=['Freeze', 'Background'])
        ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplots.png', dpi=150)
    print(f"Saved: {output_dir / 'boxplots.png'}")
    plt.close()
    
    # 3. Velocity scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    ax.scatter(df_freeze['agent_velocity'], df_freeze['target_velocity'], 
               alpha=0.5, s=20, label='Freeze', color='blue')
    ax.scatter(df_background['agent_velocity'], df_background['target_velocity'], 
               alpha=0.5, s=20, label='Background', color='orange')
    
    ax.set_xlabel('Agent Velocity (px/s)')
    ax.set_ylabel('Target Velocity (px/s)')
    ax.set_title('Agent vs Target Velocity')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Suggested thresholds
    freeze_agent_threshold = df_freeze['agent_velocity'].mean() + 1.5 * df_freeze['agent_velocity'].std()
    ax.axvline(x=freeze_agent_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Suggested threshold ({freeze_agent_threshold:.1f})')
    ax.axhline(y=freeze_agent_threshold, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity_scatter.png', dpi=150)
    print(f"Saved: {output_dir / 'velocity_scatter.png'}")
    plt.close()
    
    # Generate insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    insights = []
    
    # Velocity analysis
    freeze_vel = df_freeze['agent_velocity'].mean()
    bg_vel = df_background['agent_velocity'].mean()
    
    if freeze_vel < bg_vel:
        vel_diff_pct = ((bg_vel - freeze_vel) / bg_vel) * 100
        insights.append(f"1. Freeze has {vel_diff_pct:.1f}% lower average velocity than background")
        insights.append(f"   Freeze: {freeze_vel:.2f} px/s, Background: {bg_vel:.2f} px/s")
        
        threshold = freeze_vel + 1.5 * df_freeze['agent_velocity'].std()
        insights.append(f"   -> Suggested velocity threshold for freeze: < {threshold:.2f} px/s")
    
    # Distance analysis
    freeze_dist = df_freeze['inter_mouse_distance'].mean()
    bg_dist = df_background['inter_mouse_distance'].mean()
    
    insights.append(f"\n2. Inter-mouse distance:")
    insights.append(f"   Freeze: {freeze_dist:.2f} px, Background: {bg_dist:.2f} px")
    
    # Effect sizes
    large_effects = comparison_df[comparison_df['cohens_d'].abs() > 0.5]
    if len(large_effects) > 0:
        insights.append(f"\n3. Features with large effect sizes (|Cohen's d| > 0.5):")
        for _, row in large_effects.iterrows():
            insights.append(f"   - {row['feature']}: Cohen's d = {row['cohens_d']:.3f}")
    
    insights_text = "\n".join(insights)
    print(insights_text)
    
    # Save insights
    with open(output_dir / 'insights.txt', 'w') as f:
        f.write("FREEZE vs BACKGROUND ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(insights_text)
        f.write("\n\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS:\n")
        f.write("="*80 + "\n\n")
        if freeze_vel < bg_vel:
            threshold = freeze_vel + 1.5 * df_freeze['agent_velocity'].std()
            f.write(f"1. Add velocity gating in postprocessing:\n")
            f.write(f"   - Filter freeze predictions to require agent_velocity < {threshold:.2f} px/s\n")
            f.write(f"   - Consider target velocity as well\n\n")
        f.write("2. Increase freeze class weight in training loss\n")
        f.write("3. Add hard negative mining for background confusion\n")
        f.write("4. Consider adding explicit velocity features to model input\n")
    
    print(f"\nSaved insights to: {output_dir / 'insights.txt'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def main():
    """Main execution."""
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "kaggle"
    output_dir = Path(__file__).parent / "freeze_analysis_results"
    
    print("="*80)
    print("FREEZE vs BACKGROUND ANALYSIS")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load dataset
    print("\nLoading validation dataset...")
    try:
        dataset = V8Dataset(
            data_dir=str(data_dir),
            split='val',
            sequence_length=100,
            stride=100
        )
        print(f"Dataset loaded: {len(dataset)} sequences")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get freeze action ID
    freeze_id = ACTION_TO_ID.get('freeze', None)
    if freeze_id is None:
        print("ERROR: 'freeze' action not found in action mapping!")
        print(f"Available actions: {list(ACTION_TO_ID.keys())}")
        return
    
    print(f"Freeze action ID: {freeze_id}")
    
    # Analyze freeze class
    df_freeze = analyze_dataset(dataset, 'freeze', freeze_id, max_samples=500)
    
    if df_freeze is None or len(df_freeze) == 0:
        print("\nERROR: No freeze samples found!")
        print("This explains why the model cannot learn freeze class.")
        print("Freeze may not exist in validation data.")
        return
    
    # Analyze background class
    df_background = analyze_dataset(dataset, 'background', 0, max_samples=500)
    
    if df_background is None or len(df_background) == 0:
        print("\nERROR: No background samples found!")
        return
    
    # Compare and visualize
    compare_and_visualize(df_freeze, df_background, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
