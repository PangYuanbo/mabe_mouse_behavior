"""
Analyze differences between freeze and background classes.

This script extracts and compares motion, spatial, and temporal features
for freeze vs background frames to understand why freeze is hard to detect.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict
import sys

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

from v8_fine_grained.v8_dataset import V8Dataset


def compute_velocity(keypoints, fps=30):
    """
    Compute velocity from keypoints.
    
    Args:
        keypoints: array of shape (num_frames, num_keypoints, 2) or (num_keypoints, 2)
        fps: frames per second
        
    Returns:
        velocity in pixels per second
    """
    if len(keypoints.shape) == 2:
        # Single frame - return 0
        return 0.0
    
    # Compute frame-to-frame displacement
    displacements = np.diff(keypoints, axis=0)
    speeds = np.linalg.norm(displacements, axis=-1)  # (num_frames-1, num_keypoints)
    
    # Average speed across all keypoints
    avg_speed = np.mean(speeds)
    
    # Convert to pixels per second
    velocity = avg_speed * fps
    
    return velocity


def compute_motion_features(keypoints, fps=30):
    """
    Compute detailed motion features from keypoints.
    
    Args:
        keypoints: array of shape (num_frames, num_keypoints, 2)
        fps: frames per second
        
    Returns:
        dict of motion features
    """
    if len(keypoints.shape) == 2:
        keypoints = keypoints[np.newaxis, ...]
    
    if keypoints.shape[0] < 2:
        return {
            'velocity': 0.0,
            'acceleration': 0.0,
            'max_velocity': 0.0,
            'velocity_std': 0.0,
            'displacement': 0.0
        }
    
    # Compute velocities
    displacements = np.diff(keypoints, axis=0)
    speeds = np.linalg.norm(displacements, axis=-1)  # (num_frames-1, num_keypoints)
    frame_speeds = np.mean(speeds, axis=1) * fps  # Average across keypoints
    
    # Compute accelerations
    accelerations = np.diff(frame_speeds)
    
    # Total displacement
    total_displacement = np.sum(np.linalg.norm(displacements, axis=-1))
    
    return {
        'velocity': np.mean(frame_speeds),
        'acceleration': np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0.0,
        'max_velocity': np.max(frame_speeds),
        'velocity_std': np.std(frame_speeds),
        'displacement': total_displacement
    }


def compute_spatial_features(keypoints):
    """
    Compute spatial spread and variation features.
    
    Args:
        keypoints: array of shape (num_frames, num_keypoints, 2) or (num_keypoints, 2)
        
    Returns:
        dict of spatial features
    """
    if len(keypoints.shape) == 2:
        keypoints = keypoints[np.newaxis, ...]
    
    # Compute centroid for each frame
    centroids = np.mean(keypoints, axis=1)  # (num_frames, 2)
    
    # Compute spread (distance from centroid)
    spreads = []
    for i in range(keypoints.shape[0]):
        distances = np.linalg.norm(keypoints[i] - centroids[i], axis=1)
        spreads.append(np.mean(distances))
    
    # Compute temporal variation in centroid position
    centroid_variation = np.std(centroids, axis=0) if keypoints.shape[0] > 1 else np.array([0.0, 0.0])
    
    # Compute bounding box area
    bbox_areas = []
    for i in range(keypoints.shape[0]):
        min_coords = np.min(keypoints[i], axis=0)
        max_coords = np.max(keypoints[i], axis=0)
        area = (max_coords[0] - min_coords[0]) * (max_coords[1] - min_coords[1])
        bbox_areas.append(area)
    
    return {
        'avg_spread': np.mean(spreads),
        'spread_std': np.std(spreads),
        'centroid_variation_x': centroid_variation[0],
        'centroid_variation_y': centroid_variation[1],
        'avg_bbox_area': np.mean(bbox_areas),
        'bbox_area_std': np.std(bbox_areas)
    }


def analyze_class_characteristics(dataset, class_name, max_samples=1000, window_size=15):
    """
    Analyze characteristics of a specific class.
    
    Args:
        dataset: BehavioralMouseDataset instance
        class_name: name of class to analyze
        max_samples: maximum number of samples to analyze
        window_size: temporal window size for feature extraction
        
    Returns:
        DataFrame with features for each sample
    """
    class_idx = dataset.action_to_idx.get(class_name)
    if class_idx is None:
        print(f"Class {class_name} not found in dataset")
        return None
    
    print(f"\nAnalyzing class: {class_name} (index: {class_idx})")
    
    features_list = []
    sample_count = 0
    
    # Iterate through dataset
    for video_idx in range(len(dataset.data)):
        if sample_count >= max_samples:
            break
            
        video_data = dataset.data[video_idx]
        labels = video_data['labels']
        
        # Find frames with this class
        if class_name == 'background':
            # Background is where action_label == 0 (no action)
            class_frames = np.where(labels[:, 0] == 0)[0]
        else:
            class_frames = np.where(labels[:, 0] == class_idx)[0]
        
        if len(class_frames) == 0:
            continue
        
        # Group consecutive frames into segments
        segments = []
        if len(class_frames) > 0:
            segment_start = class_frames[0]
            for i in range(1, len(class_frames)):
                if class_frames[i] != class_frames[i-1] + 1:
                    segments.append((segment_start, class_frames[i-1]))
                    segment_start = class_frames[i]
            segments.append((segment_start, class_frames[-1]))
        
        # Analyze each segment
        for seg_start, seg_end in segments:
            if sample_count >= max_samples:
                break
            
            # Extract keypoints for this segment with context window
            context_start = max(0, seg_start - window_size // 2)
            context_end = min(len(video_data['keypoints']), seg_end + window_size // 2 + 1)
            
            segment_keypoints = video_data['keypoints'][context_start:context_end]
            
            # Get agent and target indices for this segment
            if class_name != 'background':
                agent_idx = int(labels[seg_start, 1]) if labels[seg_start, 1] > 0 else 0
                target_idx = int(labels[seg_start, 2]) if labels[seg_start, 2] > 0 else 1
            else:
                # For background, just use both mice
                agent_idx = 0
                target_idx = 1
            
            # Extract keypoints for agent and target
            agent_kps = segment_keypoints[:, agent_idx, :, :]
            target_kps = segment_keypoints[:, target_idx, :, :]
            
            # Compute motion features
            agent_motion = compute_motion_features(agent_kps)
            target_motion = compute_motion_features(target_kps)
            
            # Compute spatial features
            agent_spatial = compute_spatial_features(agent_kps)
            target_spatial = compute_spatial_features(target_kps)
            
            # Compute inter-mouse distance
            agent_centroids = np.mean(agent_kps, axis=1)
            target_centroids = np.mean(target_kps, axis=1)
            distances = np.linalg.norm(agent_centroids - target_centroids, axis=1)
            
            # Compile features
            features = {
                'class': class_name,
                'video_idx': video_idx,
                'segment_start': seg_start,
                'segment_end': seg_end,
                'segment_duration': seg_end - seg_start + 1,
                
                # Agent motion
                'agent_velocity': agent_motion['velocity'],
                'agent_acceleration': agent_motion['acceleration'],
                'agent_max_velocity': agent_motion['max_velocity'],
                'agent_velocity_std': agent_motion['velocity_std'],
                'agent_displacement': agent_motion['displacement'],
                
                # Target motion
                'target_velocity': target_motion['velocity'],
                'target_acceleration': target_motion['acceleration'],
                'target_max_velocity': target_motion['max_velocity'],
                'target_velocity_std': target_motion['velocity_std'],
                'target_displacement': target_motion['displacement'],
                
                # Agent spatial
                'agent_spread': agent_spatial['avg_spread'],
                'agent_spread_std': agent_spatial['spread_std'],
                'agent_centroid_var_x': agent_spatial['centroid_variation_x'],
                'agent_centroid_var_y': agent_spatial['centroid_variation_y'],
                'agent_bbox_area': agent_spatial['avg_bbox_area'],
                
                # Target spatial
                'target_spread': target_spatial['avg_spread'],
                'target_spread_std': target_spatial['spread_std'],
                'target_centroid_var_x': target_spatial['centroid_variation_x'],
                'target_centroid_var_y': target_spatial['centroid_variation_y'],
                'target_bbox_area': target_spatial['avg_bbox_area'],
                
                # Inter-mouse features
                'avg_distance': np.mean(distances),
                'min_distance': np.min(distances),
                'max_distance': np.max(distances),
                'distance_std': np.std(distances),
            }
            
            features_list.append(features)
            sample_count += 1
    
    print(f"Collected {sample_count} samples for class {class_name}")
    
    return pd.DataFrame(features_list)


def compare_classes(df_freeze, df_background, output_dir):
    """
    Compare freeze and background classes and generate visualizations.
    
    Args:
        df_freeze: DataFrame with freeze class features
        df_background: DataFrame with background class features
        output_dir: directory to save plots and reports
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine dataframes
    df_combined = pd.concat([df_freeze, df_background], ignore_index=True)
    
    # Select key motion features for comparison
    motion_features = [
        'agent_velocity', 'agent_max_velocity', 'agent_velocity_std',
        'target_velocity', 'target_max_velocity', 'target_velocity_std',
        'agent_displacement', 'target_displacement'
    ]
    
    spatial_features = [
        'agent_spread', 'agent_spread_std', 'agent_centroid_var_x', 'agent_centroid_var_y',
        'target_spread', 'target_spread_std', 'target_centroid_var_x', 'target_centroid_var_y',
        'avg_distance', 'distance_std'
    ]
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    summary = df_combined.groupby('class')[motion_features + spatial_features].describe()
    summary.to_csv(output_dir / 'freeze_vs_background_summary.csv')
    print(summary)
    
    # Compute statistical differences
    print("\n" + "="*80)
    print("FEATURE COMPARISON (Mean ± Std)")
    print("="*80)
    
    comparison_data = []
    for feature in motion_features + spatial_features:
        freeze_mean = df_freeze[feature].mean()
        freeze_std = df_freeze[feature].std()
        bg_mean = df_background[feature].mean()
        bg_std = df_background[feature].std()
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((freeze_std**2 + bg_std**2) / 2)
        cohens_d = (freeze_mean - bg_mean) / pooled_std if pooled_std > 0 else 0
        
        comparison_data.append({
            'feature': feature,
            'freeze_mean': freeze_mean,
            'freeze_std': freeze_std,
            'background_mean': bg_mean,
            'background_std': bg_std,
            'difference': freeze_mean - bg_mean,
            'cohens_d': cohens_d
        })
        
        print(f"{feature:30s}: Freeze={freeze_mean:8.2f}±{freeze_std:6.2f}, "
              f"Background={bg_mean:8.2f}±{bg_std:6.2f}, "
              f"Diff={freeze_mean - bg_mean:8.2f}, Cohen's d={cohens_d:6.3f}")
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'freeze_vs_background_comparison.csv', index=False)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Distribution plots for motion features
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(motion_features):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # Plot distributions
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
    plt.savefig(output_dir / 'motion_features_distribution.png', dpi=150)
    print(f"Saved: {output_dir / 'motion_features_distribution.png'}")
    plt.close()
    
    # 2. Box plots for key features
    key_features = ['agent_velocity', 'target_velocity', 'agent_max_velocity', 
                    'target_max_velocity', 'avg_distance', 'segment_duration']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(key_features):
        ax = axes[idx]
        data_to_plot = [df_freeze[feature].dropna(), df_background[feature].dropna()]
        ax.boxplot(data_to_plot, labels=['Freeze', 'Background'])
        ax.set_ylabel(feature)
        ax.set_title(feature)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'key_features_boxplot.png', dpi=150)
    print(f"Saved: {output_dir / 'key_features_boxplot.png'}")
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
    
    # Add threshold lines (example thresholds)
    ax.axvline(x=10, color='red', linestyle='--', alpha=0.7, label='Low velocity threshold')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'velocity_scatter.png', dpi=150)
    print(f"Saved: {output_dir / 'velocity_scatter.png'}")
    plt.close()
    
    # 4. Correlation heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Freeze correlation
    freeze_corr = df_freeze[motion_features[:6]].corr()
    sns.heatmap(freeze_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[0], cbar_kws={'label': 'Correlation'})
    axes[0].set_title('Freeze Class - Feature Correlations')
    
    # Background correlation
    bg_corr = df_background[motion_features[:6]].corr()
    sns.heatmap(bg_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1], cbar_kws={'label': 'Correlation'})
    axes[1].set_title('Background Class - Feature Correlations')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_heatmap.png', dpi=150)
    print(f"Saved: {output_dir / 'correlation_heatmap.png'}")
    plt.close()
    
    # 5. Generate insights report
    print("\n" + "="*80)
    print("INSIGHTS AND RECOMMENDATIONS")
    print("="*80)
    
    insights = []
    
    # Velocity analysis
    freeze_velocity = df_freeze['agent_velocity'].mean()
    bg_velocity = df_background['agent_velocity'].mean()
    
    if freeze_velocity < bg_velocity:
        velocity_diff_pct = ((bg_velocity - freeze_velocity) / bg_velocity) * 100
        insights.append(f"1. Freeze has {velocity_diff_pct:.1f}% lower average velocity than background "
                       f"({freeze_velocity:.2f} vs {bg_velocity:.2f} px/s)")
        
        # Suggest threshold
        threshold_suggestion = freeze_velocity + 1.5 * df_freeze['agent_velocity'].std()
        insights.append(f"   -> Suggested velocity threshold for freeze: < {threshold_suggestion:.2f} px/s")
    
    # Velocity variability
    freeze_vel_std = df_freeze['agent_velocity_std'].mean()
    bg_vel_std = df_background['agent_velocity_std'].mean()
    
    if freeze_vel_std < bg_vel_std:
        insights.append(f"2. Freeze shows lower velocity variability ({freeze_vel_std:.2f} vs {bg_vel_std:.2f})")
        insights.append(f"   -> Freeze involves more consistent (stable) motion patterns")
    
    # Distance analysis
    freeze_distance = df_freeze['avg_distance'].mean()
    bg_distance = df_background['avg_distance'].mean()
    
    insights.append(f"3. Average inter-mouse distance: Freeze={freeze_distance:.2f}, Background={bg_distance:.2f}")
    
    # Segment duration
    freeze_duration = df_freeze['segment_duration'].mean()
    bg_duration = df_background['segment_duration'].mean()
    
    insights.append(f"4. Average segment duration: Freeze={freeze_duration:.1f} frames, "
                   f"Background={bg_duration:.1f} frames")
    
    # Cohen's d analysis
    large_effect_features = comparison_df[comparison_df['cohens_d'].abs() > 0.5].sort_values('cohens_d', 
                                                                                               key=abs, 
                                                                                               ascending=False)
    if len(large_effect_features) > 0:
        insights.append(f"\n5. Features with large effect sizes (|Cohen's d| > 0.5):")
        for _, row in large_effect_features.head(5).iterrows():
            insights.append(f"   - {row['feature']}: Cohen's d = {row['cohens_d']:.3f}")
    
    insights_text = "\n".join(insights)
    print(insights_text)
    
    with open(output_dir / 'insights.txt', 'w') as f:
        f.write("FREEZE vs BACKGROUND ANALYSIS\n")
        f.write("="*80 + "\n\n")
        f.write(insights_text)
        f.write("\n\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS FOR MODEL IMPROVEMENT:\n")
        f.write("="*80 + "\n\n")
        f.write("1. Add velocity-based gating in postprocessing:\n")
        f.write(f"   - Filter freeze predictions to require agent_velocity < {threshold_suggestion:.2f} px/s\n")
        f.write(f"   - Consider both agent and target velocities\n\n")
        f.write("2. Add velocity stability features:\n")
        f.write("   - Use velocity std as a feature (freeze has lower variability)\n")
        f.write("   - Add temporal consistency regularization in training\n\n")
        f.write("3. Improve class separation in training:\n")
        f.write("   - Increase loss weight for freeze class (it's rare and subtle)\n")
        f.write("   - Use hard negative mining to prevent background confusion\n\n")
        f.write("4. Feature engineering:\n")
        f.write("   - Add explicit velocity and acceleration features to model input\n")
        f.write("   - Consider adding motion history features (velocity over last N frames)\n\n")
    
    print(f"\nSaved insights to: {output_dir / 'insights.txt'}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}")


def main():
    """Main execution function."""
    
    # Configuration
    data_dir = Path(__file__).parent.parent.parent / "data"
    output_dir = Path(__file__).parent / "freeze_analysis_results"
    
    print("="*80)
    print("FREEZE vs BACKGROUND ANALYSIS")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load validation dataset
    print("\nLoading validation dataset...")
    try:
        dataset = BehavioralMouseDataset(
            data_dir=str(data_dir),
            split='val',
            mode='training'
        )
        print(f"Dataset loaded: {len(dataset)} videos")
        print(f"Action classes: {dataset.action_classes}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Analyze freeze class
    print("\n" + "-"*80)
    df_freeze = analyze_class_characteristics(
        dataset, 
        class_name='freeze', 
        max_samples=500,
        window_size=15
    )
    
    if df_freeze is None or len(df_freeze) == 0:
        print("ERROR: No freeze samples found in dataset!")
        print("Checking if freeze exists in ground truth annotations...")
        
        # Debug: check if freeze exists at all
        freeze_count = 0
        for video_data in dataset.data:
            freeze_idx = dataset.action_to_idx.get('freeze', -1)
            if freeze_idx > 0:
                freeze_frames = np.sum(video_data['labels'][:, 0] == freeze_idx)
                freeze_count += freeze_frames
        
        print(f"Total freeze frames in dataset: {freeze_count}")
        
        if freeze_count == 0:
            print("Freeze class has NO samples in validation set - this explains why it cannot be learned!")
            return
    
    # Analyze background class
    print("\n" + "-"*80)
    df_background = analyze_class_characteristics(
        dataset, 
        class_name='background', 
        max_samples=500,
        window_size=15
    )
    
    if df_background is None or len(df_background) == 0:
        print("ERROR: No background samples found in dataset!")
        return
    
    # Compare classes
    print("\n" + "-"*80)
    compare_classes(df_freeze, df_background, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS PIPELINE COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
