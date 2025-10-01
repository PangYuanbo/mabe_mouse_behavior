"""
Advanced feature engineering for MABe mouse behavior detection
Based on winning solutions and research papers
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.stats import skew, kurtosis


class MouseFeatureEngineer:
    """Extract hand-crafted features from mouse keypoint data"""

    def __init__(self, num_mice=2, num_keypoints=7):
        """
        Args:
            num_mice: Number of mice in the scene
            num_keypoints: Number of keypoints per mouse
        """
        self.num_mice = num_mice
        self.num_keypoints = num_keypoints
        self.pca = None

    def reshape_keypoints(self, data):
        """
        Reshape flat keypoint data to structured format

        Args:
            data: Shape [frames, num_mice * num_keypoints * 2]

        Returns:
            reshaped: Shape [frames, num_mice, num_keypoints, 2]
        """
        frames = data.shape[0]
        return data.reshape(frames, self.num_mice, self.num_keypoints, 2)

    def compute_distances(self, keypoints):
        """
        Compute inter-mouse and intra-mouse distances

        Args:
            keypoints: [frames, num_mice, num_keypoints, 2]

        Returns:
            dict of distance features
        """
        frames = keypoints.shape[0]
        features = {}

        # Inter-mouse distances (between centroids)
        mouse0_centroid = keypoints[:, 0, :, :].mean(axis=1)  # [frames, 2]
        mouse1_centroid = keypoints[:, 1, :, :].mean(axis=1)  # [frames, 2]

        inter_dist = np.linalg.norm(mouse0_centroid - mouse1_centroid, axis=1)  # [frames]
        features['inter_mouse_distance'] = inter_dist

        # Head-to-head distance
        head0 = keypoints[:, 0, 0, :]  # Assume keypoint 0 is head
        head1 = keypoints[:, 1, 0, :]
        features['head_to_head_distance'] = np.linalg.norm(head0 - head1, axis=1)

        # Head-to-tail distances (for mounting behavior)
        tail0 = keypoints[:, 0, -1, :]  # Assume last keypoint is tail
        tail1 = keypoints[:, 1, -1, :]
        features['head0_to_tail1_distance'] = np.linalg.norm(head0 - tail1, axis=1)
        features['head1_to_tail0_distance'] = np.linalg.norm(head1 - tail0, axis=1)

        # Body length for each mouse
        for mouse_idx in range(self.num_mice):
            body_vec = keypoints[:, mouse_idx, -1, :] - keypoints[:, mouse_idx, 0, :]
            features[f'mouse{mouse_idx}_body_length'] = np.linalg.norm(body_vec, axis=1)

        return features

    def compute_velocities(self, keypoints):
        """
        Compute velocity features

        Args:
            keypoints: [frames, num_mice, num_keypoints, 2]

        Returns:
            dict of velocity features
        """
        features = {}

        # Centroid velocities
        for mouse_idx in range(self.num_mice):
            centroid = keypoints[:, mouse_idx, :, :].mean(axis=1)  # [frames, 2]
            velocity = np.diff(centroid, axis=0, prepend=centroid[0:1])  # [frames, 2]
            speed = np.linalg.norm(velocity, axis=1)  # [frames]

            features[f'mouse{mouse_idx}_speed'] = speed
            features[f'mouse{mouse_idx}_velocity_x'] = velocity[:, 0]
            features[f'mouse{mouse_idx}_velocity_y'] = velocity[:, 1]

        # Acceleration
        for mouse_idx in range(self.num_mice):
            speed = features[f'mouse{mouse_idx}_speed']
            accel = np.diff(speed, prepend=speed[0])
            features[f'mouse{mouse_idx}_acceleration'] = accel

        return features

    def compute_angles(self, keypoints):
        """
        Compute angle and orientation features

        Args:
            keypoints: [frames, num_mice, num_keypoints, 2]

        Returns:
            dict of angle features
        """
        features = {}

        # Body orientation for each mouse
        for mouse_idx in range(self.num_mice):
            head = keypoints[:, mouse_idx, 0, :]
            tail = keypoints[:, mouse_idx, -1, :]
            body_vec = head - tail

            # Orientation angle
            angles = np.arctan2(body_vec[:, 1], body_vec[:, 0])
            features[f'mouse{mouse_idx}_orientation'] = angles

            # Angular velocity
            angle_diff = np.diff(angles, prepend=angles[0])
            # Handle wraparound
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            features[f'mouse{mouse_idx}_angular_velocity'] = angle_diff

        # Relative angle between mice
        if self.num_mice == 2:
            angle0 = features['mouse0_orientation']
            angle1 = features['mouse1_orientation']
            rel_angle = angle0 - angle1
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            features['relative_orientation'] = rel_angle

        return features

    def compute_shape_features(self, keypoints):
        """
        Compute shape and posture features

        Args:
            keypoints: [frames, num_mice, num_keypoints, 2]

        Returns:
            dict of shape features
        """
        features = {}

        for mouse_idx in range(self.num_mice):
            mouse_kpts = keypoints[:, mouse_idx, :, :]  # [frames, num_keypoints, 2]

            # Compute area (convex hull approximation)
            # Use std of keypoints as proxy for spread
            spread_x = np.std(mouse_kpts[:, :, 0], axis=1)
            spread_y = np.std(mouse_kpts[:, :, 1], axis=1)
            features[f'mouse{mouse_idx}_spread'] = spread_x * spread_y

            # Ellipse ratio (elongation)
            features[f'mouse{mouse_idx}_elongation'] = spread_x / (spread_y + 1e-6)

        return features

    def compute_temporal_stats(self, features_dict, window=10):
        """
        Compute temporal statistics over a sliding window

        Args:
            features_dict: Dict of feature arrays [frames]
            window: Window size for statistics

        Returns:
            dict with temporal statistics
        """
        temporal_features = {}

        for feat_name, feat_values in features_dict.items():
            if len(feat_values.shape) != 1:
                continue  # Skip non-1D features

            # Compute rolling statistics
            frames = len(feat_values)
            for i in range(frames):
                start = max(0, i - window + 1)
                window_data = feat_values[start:i+1]

                # Initialize arrays if first iteration
                if i == 0:
                    temporal_features[f'{feat_name}_mean'] = np.zeros(frames)
                    temporal_features[f'{feat_name}_std'] = np.zeros(frames)
                    temporal_features[f'{feat_name}_min'] = np.zeros(frames)
                    temporal_features[f'{feat_name}_max'] = np.zeros(frames)

                temporal_features[f'{feat_name}_mean'][i] = np.mean(window_data)
                temporal_features[f'{feat_name}_std'][i] = np.std(window_data)
                temporal_features[f'{feat_name}_min'][i] = np.min(window_data)
                temporal_features[f'{feat_name}_max'][i] = np.max(window_data)

        return temporal_features

    def fit_pca(self, keypoints_flat, n_components=16):
        """
        Fit PCA on keypoint data

        Args:
            keypoints_flat: [num_samples, features]
            n_components: Number of PCA components
        """
        self.pca = PCA(n_components=n_components)
        self.pca.fit(keypoints_flat)
        print(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")

    def transform_pca(self, keypoints_flat):
        """
        Transform keypoint data using fitted PCA

        Args:
            keypoints_flat: [num_samples, features]

        Returns:
            pca_features: [num_samples, n_components]
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        return self.pca.transform(keypoints_flat)

    def extract_all_features(self, data, include_pca=True, include_temporal=True):
        """
        Extract all features from raw keypoint data

        Args:
            data: Raw keypoint data [frames, num_mice * num_keypoints * 2]
            include_pca: Whether to include PCA features
            include_temporal: Whether to include temporal statistics

        Returns:
            feature_array: [frames, num_features]
        """
        # Reshape keypoints
        keypoints = self.reshape_keypoints(data)

        # Extract all feature groups
        all_features = {}
        all_features.update(self.compute_distances(keypoints))
        all_features.update(self.compute_velocities(keypoints))
        all_features.update(self.compute_angles(keypoints))
        all_features.update(self.compute_shape_features(keypoints))

        # Optionally add temporal statistics
        if include_temporal:
            temporal_feats = self.compute_temporal_stats(all_features, window=10)
            all_features.update(temporal_feats)

        # Stack all features
        feature_list = []
        for feat_name in sorted(all_features.keys()):
            feature_list.append(all_features[feat_name].reshape(-1, 1))

        features = np.hstack(feature_list)  # [frames, num_hand_crafted_features]

        # Optionally add PCA features
        if include_pca and self.pca is not None:
            pca_feats = self.transform_pca(data)
            features = np.hstack([features, pca_feats])

        return features.astype(np.float32)