"""
V8.6 MARS-Inspired Feature Engineering
=======================================
Implements rich feature extraction based on MARS paper (eLife 2021)

Features added (270+ dimensions total):
1. Joint angles (12 angles per mouse = 48 total)
2. Relative positions between mice (28 features per pair)
3. Body orientation (4 features per mouse = 16 total)
4. Inter-mouse distances (7 bodyparts × 6 pairs = 42 features)
5. Motion derivatives (jerk, angular velocity)
6. Social interaction features (contact zones, relative velocity)
"""

import numpy as np
from typing import Tuple


class MARSFeatureExtractor:
    """
    Extract MARS-style features from raw keypoints

    Input: [T, 56] raw keypoints (7 bodyparts × 4 mice × 2 coords)
    Output: [T, D] enriched features (D ≈ 350-400)
    """

    def __init__(self, fps: float = 33.3):
        self.fps = fps
        self.dt = 1.0 / fps

        # Bodypart indices (in order)
        self.bodyparts = ['nose', 'ear_left', 'ear_right', 'neck',
                          'hip_left', 'hip_right', 'tail_base']
        self.num_bodyparts = 7
        self.num_mice = 4

    def extract_all_features(self, keypoints: np.ndarray) -> np.ndarray:
        """
        Extract all MARS-inspired features

        Args:
            keypoints: [T, 56] raw coordinates

        Returns:
            features: [T, ~370] enriched features
        """
        T = keypoints.shape[0]

        # Reshape to [T, 4 mice, 7 bodyparts, 2 coords]
        kp = keypoints.reshape(T, self.num_mice, self.num_bodyparts, 2)

        feature_groups = []

        # 1. Raw coordinates (56)
        feature_groups.append(keypoints)

        # 2. Joint angles per mouse (48 = 12 angles × 4 mice)
        joint_angles = self._compute_joint_angles(kp)
        feature_groups.append(joint_angles)

        # 3. Body orientation per mouse (16 = 4 angles × 4 mice)
        body_orientation = self._compute_body_orientation(kp)
        feature_groups.append(body_orientation)

        # 4. Velocity and speed per bodypart (56 = 28 keypoints × 2)
        velocity, speed = self._compute_velocity(kp)
        feature_groups.append(speed)  # [T, 28]

        # 5. Acceleration per bodypart (28)
        acceleration = self._compute_acceleration(velocity)
        feature_groups.append(acceleration)

        # 6. Jerk (rate of acceleration change) per bodypart (28)
        jerk = self._compute_jerk(acceleration)
        feature_groups.append(jerk)

        # 7. Angular velocity per mouse (4)
        angular_velocity = self._compute_angular_velocity(body_orientation)
        feature_groups.append(angular_velocity)

        # 8. Inter-mouse distances (42 = 7 bodyparts × 6 pairs)
        inter_mouse_dist = self._compute_inter_mouse_distances(kp)
        feature_groups.append(inter_mouse_dist)

        # 9. Relative positions between mice (48 = 12 features × 4 pairs)
        relative_pos = self._compute_relative_positions(kp)
        feature_groups.append(relative_pos)

        # 10. Social interaction features (24)
        social_features = self._compute_social_features(kp, velocity)
        feature_groups.append(social_features)

        # Concatenate all features
        all_features = np.concatenate(feature_groups, axis=1)

        return all_features

    def _compute_joint_angles(self, kp: np.ndarray) -> np.ndarray:
        """
        Compute joint angles for each mouse

        Angles computed:
        - Ear-left to neck to ear-right
        - Nose to neck to tail_base (body angle)
        - Neck to hip_left to tail_base
        - Neck to hip_right to tail_base
        - And more...

        Args:
            kp: [T, 4, 7, 2]

        Returns:
            angles: [T, 48] (12 angles per mouse × 4 mice)
        """
        T = kp.shape[0]
        angles_list = []

        # Bodypart name to index mapping
        bp_idx = {name: i for i, name in enumerate(self.bodyparts)}

        for mouse_id in range(self.num_mice):
            mouse_kp = kp[:, mouse_id, :, :]  # [T, 7, 2]

            # Angle 1: ear_left -> neck -> ear_right
            angle1 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['ear_left']] - mouse_kp[:, bp_idx['neck']],
                mouse_kp[:, bp_idx['ear_right']] - mouse_kp[:, bp_idx['neck']]
            )

            # Angle 2: nose -> neck -> tail_base (spine angle)
            angle2 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['neck']],
                mouse_kp[:, bp_idx['tail_base']] - mouse_kp[:, bp_idx['neck']]
            )

            # Angle 3: neck -> hip_left -> tail_base
            angle3 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['neck']] - mouse_kp[:, bp_idx['hip_left']],
                mouse_kp[:, bp_idx['tail_base']] - mouse_kp[:, bp_idx['hip_left']]
            )

            # Angle 4: neck -> hip_right -> tail_base
            angle4 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['neck']] - mouse_kp[:, bp_idx['hip_right']],
                mouse_kp[:, bp_idx['tail_base']] - mouse_kp[:, bp_idx['hip_right']]
            )

            # Angle 5: hip_left -> neck -> hip_right
            angle5 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['hip_left']] - mouse_kp[:, bp_idx['neck']],
                mouse_kp[:, bp_idx['hip_right']] - mouse_kp[:, bp_idx['neck']]
            )

            # Angle 6: nose -> neck -> ear_left
            angle6 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['neck']],
                mouse_kp[:, bp_idx['ear_left']] - mouse_kp[:, bp_idx['neck']]
            )

            # Angle 7: nose -> neck -> ear_right
            angle7 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['neck']],
                mouse_kp[:, bp_idx['ear_right']] - mouse_kp[:, bp_idx['neck']]
            )

            # Angle 8: hip_left -> tail_base -> hip_right
            angle8 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['hip_left']] - mouse_kp[:, bp_idx['tail_base']],
                mouse_kp[:, bp_idx['hip_right']] - mouse_kp[:, bp_idx['tail_base']]
            )

            # Angle 9-12: Additional geometric relationships
            angle9 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['hip_left']],
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['hip_right']]
            )

            angle10 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['ear_left']] - mouse_kp[:, bp_idx['hip_left']],
                mouse_kp[:, bp_idx['ear_right']] - mouse_kp[:, bp_idx['hip_right']]
            )

            angle11 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['tail_base']],
                mouse_kp[:, bp_idx['neck']] - mouse_kp[:, bp_idx['tail_base']]
            )

            angle12 = self._angle_between_vectors(
                mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['neck']],
                mouse_kp[:, bp_idx['hip_left']] - mouse_kp[:, bp_idx['hip_right']]
            )

            # Stack all angles for this mouse
            mouse_angles = np.stack([
                angle1, angle2, angle3, angle4, angle5, angle6,
                angle7, angle8, angle9, angle10, angle11, angle12
            ], axis=1)  # [T, 12]

            angles_list.append(mouse_angles)

        # Concatenate all mice: [T, 48]
        all_angles = np.concatenate(angles_list, axis=1)
        return all_angles

    def _compute_body_orientation(self, kp: np.ndarray) -> np.ndarray:
        """
        Compute body orientation angle for each mouse

        Orientation: angle from tail_base -> nose direction relative to x-axis

        Args:
            kp: [T, 4, 7, 2]

        Returns:
            orientations: [T, 4] angles in radians
        """
        T = kp.shape[0]
        bp_idx = {name: i for i, name in enumerate(self.bodyparts)}

        orientations = []

        for mouse_id in range(self.num_mice):
            mouse_kp = kp[:, mouse_id, :, :]  # [T, 7, 2]

            # Direction vector from tail_base to nose
            direction = mouse_kp[:, bp_idx['nose']] - mouse_kp[:, bp_idx['tail_base']]

            # Angle relative to x-axis
            angle = np.arctan2(direction[:, 1], direction[:, 0])  # [T]

            # Also compute sine and cosine for circular features
            angle_sin = np.sin(angle)
            angle_cos = np.cos(angle)

            orientations.append(angle)
            orientations.append(angle_sin)
            orientations.append(angle_cos)

        # Return [T, 12] (angle, sin, cos) × 4 mice
        # But we simplify to [T, 4] for now
        return np.stack(orientations[:self.num_mice], axis=1)

    def _compute_velocity(self, kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocity and speed for each keypoint

        Args:
            kp: [T, 4, 7, 2]

        Returns:
            velocity: [T, 4, 7, 2] velocity vectors
            speed: [T, 28] speed magnitudes
        """
        T = kp.shape[0]

        # Velocity: displacement / dt
        velocity = np.zeros_like(kp)
        if T > 1:
            velocity[1:] = (kp[1:] - kp[:-1]) / self.dt
            velocity[0] = velocity[1]  # Copy first frame

        # Speed magnitude
        speed = np.sqrt(np.sum(velocity ** 2, axis=-1))  # [T, 4, 7]
        speed = speed.reshape(T, -1)  # [T, 28]

        return velocity, speed

    def _compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute acceleration magnitude for each keypoint

        Args:
            velocity: [T, 4, 7, 2]

        Returns:
            acceleration: [T, 28]
        """
        T = velocity.shape[0]

        accel_vec = np.zeros_like(velocity)
        if T > 1:
            accel_vec[1:] = (velocity[1:] - velocity[:-1]) / self.dt
            accel_vec[0] = accel_vec[1]

        accel_mag = np.sqrt(np.sum(accel_vec ** 2, axis=-1))  # [T, 4, 7]
        accel_mag = accel_mag.reshape(T, -1)  # [T, 28]

        return accel_mag

    def _compute_jerk(self, acceleration: np.ndarray) -> np.ndarray:
        """
        Compute jerk (rate of change of acceleration) - critical for freeze detection

        Args:
            acceleration: [T, 28]

        Returns:
            jerk: [T, 28]
        """
        T = acceleration.shape[0]

        jerk = np.zeros_like(acceleration)
        if T > 1:
            jerk[1:] = (acceleration[1:] - acceleration[:-1]) / self.dt
            jerk[0] = jerk[1]

        return jerk

    def _compute_angular_velocity(self, orientation: np.ndarray) -> np.ndarray:
        """
        Compute angular velocity (how fast the mouse is turning)

        Args:
            orientation: [T, 4] body angles

        Returns:
            angular_vel: [T, 4]
        """
        T = orientation.shape[0]

        angular_vel = np.zeros_like(orientation)
        if T > 1:
            # Handle angle wrapping at ±π
            angle_diff = np.diff(orientation, axis=0)
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            angular_vel[1:] = angle_diff / self.dt
            angular_vel[0] = angular_vel[1]

        return angular_vel

    def _compute_inter_mouse_distances(self, kp: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between mice for all bodyparts

        For each bodypart, compute distance between all mouse pairs:
        - Mouse 1-2, 1-3, 1-4, 2-3, 2-4, 3-4 (6 pairs)

        Args:
            kp: [T, 4, 7, 2]

        Returns:
            distances: [T, 42] (7 bodyparts × 6 pairs)
        """
        T = kp.shape[0]
        distances_list = []

        # All mouse pairs
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

        for bp_idx in range(self.num_bodyparts):
            for (m1, m2) in pairs:
                pos1 = kp[:, m1, bp_idx, :]  # [T, 2]
                pos2 = kp[:, m2, bp_idx, :]  # [T, 2]
                dist = np.sqrt(np.sum((pos1 - pos2) ** 2, axis=1))  # [T]
                distances_list.append(dist)

        distances = np.stack(distances_list, axis=1)  # [T, 42]
        return distances

    def _compute_relative_positions(self, kp: np.ndarray) -> np.ndarray:
        """
        Compute relative position features between key mouse pairs

        Focus on agent-target relationships (assuming consecutive pairs):
        - Mouse 1-2, 2-3, 3-4, 1-3 (4 relevant pairs for social behaviors)

        For each pair, compute:
        - Relative x, y (nose-to-nose)
        - Relative x, y (nose-to-tail)
        - Distance nose-to-nose
        - Distance nose-to-tail
        - Distance nose-to-body (closest point)
        - Relative angle (facing direction)

        Args:
            kp: [T, 4, 7, 2]

        Returns:
            relative_features: [T, 48] (12 features × 4 pairs)
        """
        T = kp.shape[0]
        bp_idx = {name: i for i, name in enumerate(self.bodyparts)}

        # Key pairs for social interactions
        pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]

        features_list = []

        for (m1, m2) in pairs:
            # Get key bodyparts
            m1_nose = kp[:, m1, bp_idx['nose'], :]
            m1_tail = kp[:, m1, bp_idx['tail_base'], :]
            m1_neck = kp[:, m1, bp_idx['neck'], :]

            m2_nose = kp[:, m2, bp_idx['nose'], :]
            m2_tail = kp[:, m2, bp_idx['tail_base'], :]
            m2_neck = kp[:, m2, bp_idx['neck'], :]

            # 1-2: Relative position (nose-to-nose)
            rel_pos_nn = m2_nose - m1_nose  # [T, 2]

            # 3-4: Relative position (nose-to-tail)
            rel_pos_nt = m2_tail - m1_nose  # [T, 2]

            # 5: Distance nose-to-nose
            dist_nn = np.sqrt(np.sum(rel_pos_nn ** 2, axis=1, keepdims=True))  # [T, 1]

            # 6: Distance nose-to-tail
            dist_nt = np.sqrt(np.sum(rel_pos_nt ** 2, axis=1, keepdims=True))  # [T, 1]

            # 7: Distance nose-to-neck
            dist_neck = np.sqrt(np.sum((m2_neck - m1_nose) ** 2, axis=1, keepdims=True))  # [T, 1]

            # 8-9: Relative orientation (direction from m1 to m2)
            m1_direction = m1_nose - m1_tail  # [T, 2]
            m2_direction = m2_nose - m2_tail  # [T, 2]

            m1_angle = np.arctan2(m1_direction[:, 1:2], m1_direction[:, 0:1])  # [T, 1]
            m2_angle = np.arctan2(m2_direction[:, 1:2], m2_direction[:, 0:1])  # [T, 1]

            # 10: Angle difference (are they facing same direction?)
            angle_diff = m2_angle - m1_angle
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # [T, 1]

            # 11-12: Is m1 facing m2? Is m2 facing m1?
            vec_to_m2 = m2_nose - m1_nose
            angle_to_m2 = np.arctan2(vec_to_m2[:, 1:2], vec_to_m2[:, 0:1])
            facing_score = np.cos(angle_to_m2 - m1_angle)  # [T, 1]

            vec_to_m1 = m1_nose - m2_nose
            angle_to_m1 = np.arctan2(vec_to_m1[:, 1:2], vec_to_m1[:, 0:1])
            facing_back_score = np.cos(angle_to_m1 - m2_angle)  # [T, 1]

            # Concatenate all features for this pair
            pair_features = np.concatenate([
                rel_pos_nn,           # [T, 2]
                rel_pos_nt,           # [T, 2]
                dist_nn,              # [T, 1]
                dist_nt,              # [T, 1]
                dist_neck,            # [T, 1]
                m1_angle,             # [T, 1]
                m2_angle,             # [T, 1]
                angle_diff,           # [T, 1]
                facing_score,         # [T, 1]
                facing_back_score     # [T, 1]
            ], axis=1)  # [T, 12]

            features_list.append(pair_features)

        # Concatenate all pairs: [T, 48]
        relative_features = np.concatenate(features_list, axis=1)
        return relative_features

    def _compute_social_features(self, kp: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """
        Compute social interaction specific features

        Features:
        - Contact detection (distance < threshold)
        - Relative velocity (approaching vs retreating)
        - Alignment (parallel movement)

        Args:
            kp: [T, 4, 7, 2]
            velocity: [T, 4, 7, 2]

        Returns:
            social_features: [T, 24] (6 features × 4 pairs)
        """
        T = kp.shape[0]
        bp_idx = {name: i for i, name in enumerate(self.bodyparts)}

        # Key pairs
        pairs = [(0, 1), (1, 2), (2, 3), (0, 2)]

        features_list = []

        for (m1, m2) in pairs:
            # Positions
            m1_nose = kp[:, m1, bp_idx['nose'], :]
            m2_nose = kp[:, m2, bp_idx['nose'], :]
            m2_neck = kp[:, m2, bp_idx['neck'], :]
            m2_tail = kp[:, m2, bp_idx['tail_base'], :]

            # Velocities
            m1_vel = velocity[:, m1, bp_idx['nose'], :]  # [T, 2]
            m2_vel = velocity[:, m2, bp_idx['nose'], :]  # [T, 2]

            # 1: Close contact (nose-to-body distance)
            min_dist_to_body = np.min([
                np.sqrt(np.sum((m1_nose - m2_nose) ** 2, axis=1)),
                np.sqrt(np.sum((m1_nose - m2_neck) ** 2, axis=1)),
                np.sqrt(np.sum((m1_nose - m2_tail) ** 2, axis=1))
            ], axis=0)  # [T]

            contact_indicator = (min_dist_to_body < 50).astype(np.float32)  # [T]

            # 2: Relative velocity magnitude
            rel_vel = m2_vel - m1_vel  # [T, 2]
            rel_speed = np.sqrt(np.sum(rel_vel ** 2, axis=1))  # [T]

            # 3: Approaching vs retreating
            vec_between = m2_nose - m1_nose  # [T, 2]
            dist_between = np.sqrt(np.sum(vec_between ** 2, axis=1, keepdims=True))  # [T, 1]

            # Dot product: negative = approaching, positive = retreating
            approach_score = np.sum(rel_vel * vec_between, axis=1) / (dist_between.squeeze() + 1e-6)  # [T]

            # 4: Movement alignment (parallel movement)
            m1_speed = np.sqrt(np.sum(m1_vel ** 2, axis=1, keepdims=True))  # [T, 1]
            m2_speed = np.sqrt(np.sum(m2_vel ** 2, axis=1, keepdims=True))  # [T, 1]

            alignment = np.sum(m1_vel * m2_vel, axis=1) / (m1_speed.squeeze() * m2_speed.squeeze() + 1e-6)  # [T]

            # 5-6: Distance and its rate of change
            dist_change = np.zeros(T)
            if T > 1:
                dist_change[1:] = (dist_between[1:, 0] - dist_between[:-1, 0]) / self.dt
                dist_change[0] = dist_change[1]

            # Stack features for this pair
            pair_features = np.stack([
                contact_indicator,
                rel_speed,
                approach_score,
                alignment,
                dist_between.squeeze(),
                dist_change
            ], axis=1)  # [T, 6]

            features_list.append(pair_features)

        # Concatenate all pairs: [T, 24]
        social_features = np.concatenate(features_list, axis=1)
        return social_features

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """
        Compute angle between two vectors

        Args:
            v1, v2: [T, 2] vectors

        Returns:
            angle: [T] angles in radians
        """
        # Normalize vectors
        v1_norm = np.sqrt(np.sum(v1 ** 2, axis=1, keepdims=True)) + 1e-8
        v2_norm = np.sqrt(np.sum(v2 ** 2, axis=1, keepdims=True)) + 1e-8

        v1_unit = v1 / v1_norm
        v2_unit = v2 / v2_norm

        # Dot product
        dot_product = np.sum(v1_unit * v2_unit, axis=1)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Angle
        angle = np.arccos(dot_product)

        return angle
