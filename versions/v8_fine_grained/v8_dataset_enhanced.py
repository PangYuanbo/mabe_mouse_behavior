"""
V8 Dataset Enhanced - 添加更丰富的运动统计特征
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from .action_mapping import ACTION_TO_ID


class V8DatasetEnhanced(Dataset):
    """
    V8 Dataset 增强版，添加更丰富的运动特征
    
    特征结构：
    - 56: 原始坐标 (7 bodyparts * 4 mice * 2 coords)
    - 28: 每个关键点的速度
    - 28: 每个关键点的加速度
    - 8: 每只老鼠的平均速度 (4 mice * 2: mean + max)
    - 8: 每只老鼠的平均加速度 (4 mice * 2: mean + max)
    - 4: 每只老鼠的速度标准差 (运动稳定性)
    - 6: 老鼠之间的相对距离 (C(4,2) = 6 pairs)
    - 6: 老鼠之间的相对速度
    
    总计：144 维特征
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 100,
        stride: int = 25,
        fps: float = 33.3
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.stride = stride if split == 'train' else sequence_length
        self.fps = fps

        # Load metadata
        csv_path = self.data_dir / f"{split}.csv" if (self.data_dir / f"{split}.csv").exists() else self.data_dir / "train.csv"
        metadata = pd.read_csv(csv_path)

        # Filter to train/val split
        np.random.seed(42)
        video_ids = metadata['video_id'].unique()
        np.random.shuffle(video_ids)
        split_idx = int(len(video_ids) * 0.8)

        if split == 'train':
            selected_ids = video_ids[:split_idx]
        else:
            selected_ids = video_ids[split_idx:]

        metadata = metadata[metadata['video_id'].isin(selected_ids)]

        # Filter to labs with annotations
        labs_with_annotations = [
            'AdaptableSnail', 'BoisterousParrot', 'CRIM13', 'CalMS21_supplemental',
            'CalMS21_task1', 'CalMS21_task2', 'CautiousGiraffe', 'DeliriousFly',
            'ElegantMink', 'GroovyShrew', 'InvincibleJellyfish', 'JovialSwallow',
            'LyricalHare', 'NiftyGoldfinch', 'PleasantMeerkat', 'ReflectiveManatee',
            'SparklingTapir', 'TranquilPanther', 'UppityFerret'
        ]
        metadata = metadata[metadata['lab_id'].isin(labs_with_annotations)]

        self.metadata = metadata
        self.sequences = []

        print(f"Loading {split} data (Enhanced) from {data_dir}...")
        self._load_sequences()

        print(f"[OK] Loaded {len(self.sequences)} sequences for {split} (Enhanced)")

    def _load_sequences(self):
        """Load all video sequences and create sliding windows"""
        from tqdm import tqdm

        failed_count = 0

        for idx, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc=f"Processing {self.split} videos"):
            video_id = row['video_id']
            lab_id = row['lab_id']

            tracking_file = self.data_dir / "train_tracking" / lab_id / f"{video_id}.parquet"
            annotation_file = self.data_dir / "train_annotation" / lab_id / f"{video_id}.parquet"

            if not tracking_file.exists():
                failed_count += 1
                continue

            # Process video
            keypoints, action_labels, agent_labels, target_labels = self._process_video(
                tracking_file, annotation_file
            )

            if keypoints is None or len(keypoints) < self.sequence_length:
                failed_count += 1
                continue

            # Create sliding windows
            num_frames = len(keypoints)
            for start_idx in range(0, num_frames - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length

                self.sequences.append({
                    'keypoints': keypoints[start_idx:end_idx],
                    'action': action_labels[start_idx:end_idx],
                    'agent': agent_labels[start_idx:end_idx],
                    'target': target_labels[start_idx:end_idx],
                })

        if failed_count > 0:
            print(f"  [!] {failed_count} videos failed to load")

    def _process_video(
        self,
        tracking_file: Path,
        annotation_file: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process a single video

        Returns:
            keypoints: [T, 144] with enhanced motion features
            action_labels: [T] action class IDs
            agent_labels: [T] agent mouse IDs (0-3)
            target_labels: [T] target mouse IDs (0-3)
        """
        try:
            # Load tracking data
            tracking_df = pd.read_parquet(tracking_file)

            # Standard bodyparts (matching actual data naming)
            standard_bodyparts = [
                'nose', 'ear_left', 'ear_right', 'neck',
                'hip_left', 'hip_right', 'tail_base'
            ]

            # Filter to standard bodyparts
            tracking_df = tracking_df[tracking_df['bodypart'].isin(standard_bodyparts)]

            # Check if we have valid data
            if len(tracking_df) == 0 or tracking_df['video_frame'].isna().all():
                return None, None, None, None

            # Create fixed-size keypoint array using pivot
            max_frame = tracking_df['video_frame'].max()
            if pd.isna(max_frame):
                return None, None, None, None

            num_frames = int(max_frame) + 1
            num_mice = 4
            num_bodyparts = len(standard_bodyparts)

            # Pivot x and y separately
            x_pivot = tracking_df.pivot_table(
                index='video_frame',
                columns=['mouse_id', 'bodypart'],
                values='x',
                aggfunc='first'
            )
            y_pivot = tracking_df.pivot_table(
                index='video_frame',
                columns=['mouse_id', 'bodypart'],
                values='y',
                aggfunc='first'
            )

            # Initialize output array - fixed size [T, 56]
            # 7 bodyparts * 4 mice * 2 coords = 56
            keypoints_raw = np.zeros((num_frames, num_mice * num_bodyparts * 2), dtype=np.float32)

            # Fill in data
            for mouse_id in range(1, 5):
                for bp_idx, bodypart in enumerate(standard_bodyparts):
                    if (mouse_id, bodypart) in x_pivot.columns:
                        frames = x_pivot.index.values.astype(int)
                        x_vals = x_pivot[(mouse_id, bodypart)].values
                        y_vals = y_pivot[(mouse_id, bodypart)].values

                        base_idx = (mouse_id - 1) * num_bodyparts * 2 + bp_idx * 2
                        keypoints_raw[frames, base_idx] = x_vals
                        keypoints_raw[frames, base_idx + 1] = y_vals

            # Now keypoints_raw is guaranteed to be [T, 56]
            keypoints = np.nan_to_num(keypoints_raw, nan=0.0)

            # Initialize labels
            action_labels = np.zeros(num_frames, dtype=np.int64)  # 0 = background
            agent_labels = np.zeros(num_frames, dtype=np.int64)   # Default mouse1
            target_labels = np.zeros(num_frames, dtype=np.int64)  # Default mouse1

            # Load annotations
            if annotation_file.exists():
                annotation_df = pd.read_parquet(annotation_file)

                for _, row in annotation_df.iterrows():
                    action = row['action']
                    start_frame = row['start_frame']
                    stop_frame = row['stop_frame']
                    agent_id = row.get('agent_id', 1)  # Default to mouse1
                    target_id = row.get('target_id', 2)  # Default to mouse2

                    # Map action to ID
                    action_id = ACTION_TO_ID.get(action, 0)

                    # Map agent/target (1-4 in data -> 0-3 for model)
                    agent_idx = int(agent_id) - 1 if isinstance(agent_id, (int, float)) else 0
                    target_idx = int(target_id) - 1 if isinstance(target_id, (int, float)) else 1

                    agent_idx = np.clip(agent_idx, 0, 3)
                    target_idx = np.clip(target_idx, 0, 3)

                    # Fill frames
                    for frame in range(start_frame, min(stop_frame + 1, num_frames)):
                        if action_id > action_labels[frame]:
                            action_labels[frame] = action_id
                            agent_labels[frame] = agent_idx
                            target_labels[frame] = target_idx

            # Add enhanced motion features
            keypoints = self._add_enhanced_motion_features(keypoints, self.fps)

            return keypoints, action_labels, agent_labels, target_labels

        except Exception as e:
            print(f"Error processing video: {e}")
            return None, None, None, None

    def _add_enhanced_motion_features(self, keypoints: np.ndarray, fps: float = 33.3) -> np.ndarray:
        """
        Add enhanced motion features including per-mouse statistics
        
        Input: [T, 56] (7 bodyparts * 4 mice * 2 coords)
        Output: [T, 144] (56 coords + 28 speeds + 28 accels + 32 aggregated features)
        """
        dt = 1.0 / fps
        T, D = keypoints.shape

        # D should always be 56
        assert D == 56, f"Expected 56 coords, got {D}"

        num_keypoints = D // 2  # 28 keypoints (7 per mouse * 4 mice)
        coords = keypoints.reshape(T, num_keypoints, 2)  # [T, 28, 2]

        # ========== 基础运动特征 ==========
        # Velocity vectors
        velocity = np.zeros_like(coords)
        if T > 1:
            velocity[1:] = (coords[1:] - coords[:-1]) / dt
            velocity[0] = velocity[1]

        # Speed magnitude per keypoint
        speed = np.sqrt(np.sum(velocity ** 2, axis=2, keepdims=True))  # [T, 28, 1]

        # Acceleration vectors
        acceleration_vec = np.zeros_like(velocity)
        if T > 1:
            acceleration_vec[1:] = (velocity[1:] - velocity[:-1]) / dt
            acceleration_vec[0] = acceleration_vec[1]

        acceleration = np.sqrt(np.sum(acceleration_vec ** 2, axis=2, keepdims=True))  # [T, 28, 1]

        # ========== 每只老鼠的聚合运动特征 ==========
        num_bodyparts_per_mouse = 7
        mouse_features = []

        for mouse_idx in range(4):
            start_kp = mouse_idx * num_bodyparts_per_mouse
            end_kp = start_kp + num_bodyparts_per_mouse

            # 这只老鼠的所有关键点速度
            mouse_speed = speed[:, start_kp:end_kp, 0]  # [T, 7]
            mouse_accel = acceleration[:, start_kp:end_kp, 0]  # [T, 7]

            # 统计特征
            mean_speed = np.mean(mouse_speed, axis=1, keepdims=True)  # [T, 1]
            max_speed = np.max(mouse_speed, axis=1, keepdims=True)    # [T, 1]
            std_speed = np.std(mouse_speed, axis=1, keepdims=True)    # [T, 1] - 运动稳定性

            mean_accel = np.mean(mouse_accel, axis=1, keepdims=True)  # [T, 1]
            max_accel = np.max(mouse_accel, axis=1, keepdims=True)    # [T, 1]

            mouse_features.append(mean_speed)
            mouse_features.append(max_speed)
            mouse_features.append(std_speed)
            mouse_features.append(mean_accel)
            mouse_features.append(max_accel)

        # mouse_features: 4 mice * 5 features = 20 features

        # ========== 老鼠之间的相对运动特征 ==========
        # 计算每只老鼠的质心
        mouse_centroids = []
        for mouse_idx in range(4):
            start_kp = mouse_idx * num_bodyparts_per_mouse
            end_kp = start_kp + num_bodyparts_per_mouse
            centroid = np.mean(coords[:, start_kp:end_kp, :], axis=1)  # [T, 2]
            mouse_centroids.append(centroid)

        mouse_centroids = np.stack(mouse_centroids, axis=1)  # [T, 4, 2]

        # 计算每对老鼠之间的距离和相对速度
        pairwise_features = []
        for i in range(4):
            for j in range(i + 1, 4):
                # 距离
                distance = np.linalg.norm(
                    mouse_centroids[:, i, :] - mouse_centroids[:, j, :],
                    axis=1, keepdims=True
                )  # [T, 1]

                # 相对速度（距离变化率）
                relative_velocity = np.zeros((T, 1), dtype=np.float32)
                if T > 1:
                    relative_velocity[1:] = (distance[1:] - distance[:-1]) / dt
                    relative_velocity[0] = relative_velocity[1]

                pairwise_features.append(distance)
                pairwise_features.append(relative_velocity)

        # pairwise_features: C(4,2) * 2 = 6 * 2 = 12 features

        # ========== 拼接所有特征 ==========
        keypoints_flat = coords.reshape(T, -1)  # [T, 56]
        speed_flat = speed.squeeze(-1)          # [T, 28]
        accel_flat = acceleration.squeeze(-1)   # [T, 28]
        mouse_features_flat = np.concatenate(mouse_features, axis=1)  # [T, 20]
        pairwise_features_flat = np.concatenate(pairwise_features, axis=1)  # [T, 12]

        enhanced = np.concatenate([
            keypoints_flat,          # 56
            speed_flat,              # 28
            accel_flat,              # 28
            mouse_features_flat,     # 20
            pairwise_features_flat   # 12
        ], axis=1)  # [T, 144]

        return enhanced

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        keypoints = torch.FloatTensor(seq['keypoints'])
        action = torch.LongTensor(seq['action'])
        agent = torch.LongTensor(seq['agent'])
        target = torch.LongTensor(seq['target'])

        return keypoints, action, agent, target


def create_v8_enhanced_dataloaders(
    data_dir: str,
    batch_size: int = 256,
    sequence_length: int = 100,
    stride: int = 25,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create V8 Enhanced train and validation dataloaders

    Returns:
        train_loader, val_loader
    """
    train_dataset = V8DatasetEnhanced(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        stride=stride
    )

    val_dataset = V8DatasetEnhanced(
        data_dir=data_dir,
        split='val',
        sequence_length=sequence_length,
        stride=sequence_length  # No overlap for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader
