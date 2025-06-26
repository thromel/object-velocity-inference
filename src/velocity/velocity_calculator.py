"""
Calculates real-world velocities from pixel coordinates.
Uses camera calibration and temporal smoothing for accuracy.
"""

import numpy as np
import cv2
import json
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import time
from scipy.spatial.distance import cdist
from scipy.optimize import minimize


@dataclass
class VelocityMeasurement:
    """
    Data class for velocity measurements
    """
    track_id: int
    velocity_kmh: float
    velocity_ms: float
    world_position: Tuple[float, float]
    direction_angle: float  # Angle in degrees
    confidence: float
    timestamp: float
    frame_number: int


@dataclass
class TrackHistory:
    """
    History data for a single track
    """
    positions_image: deque  # Image coordinates
    positions_world: deque  # World coordinates
    timestamps: deque
    frame_numbers: deque
    velocities: deque
    last_update: float
    
    def __post_init__(self):
        if not isinstance(self.positions_image, deque):
            self.positions_image = deque(self.positions_image, maxlen=self.positions_image.maxlen if hasattr(self.positions_image, 'maxlen') else None)
        if not isinstance(self.positions_world, deque):
            self.positions_world = deque(self.positions_world, maxlen=self.positions_world.maxlen if hasattr(self.positions_world, 'maxlen') else None)
        if not isinstance(self.timestamps, deque):
            self.timestamps = deque(self.timestamps, maxlen=self.timestamps.maxlen if hasattr(self.timestamps, 'maxlen') else None)
        if not isinstance(self.frame_numbers, deque):
            self.frame_numbers = deque(self.frame_numbers, maxlen=self.frame_numbers.maxlen if hasattr(self.frame_numbers, 'maxlen') else None)
        if not isinstance(self.velocities, deque):
            self.velocities = deque(self.velocities, maxlen=self.velocities.maxlen if hasattr(self.velocities, 'maxlen') else None)


class CameraCalibration:
    """
    Camera calibration utilities for coordinate transformations
    """
    
    def __init__(self, calibration_data: Dict[str, Any]):
        """
        Initialize calibration from data dictionary
        
        Args:
            calibration_data: Dictionary containing calibration parameters
        """
        self.camera_matrix = np.array(calibration_data.get('camera_matrix', np.eye(3)))
        self.dist_coeffs = np.array(calibration_data.get('dist_coeffs', [0, 0, 0, 0, 0]))
        self.homography_matrix = np.array(calibration_data.get('homography_matrix', np.eye(3)))
        self.image_size = tuple(calibration_data.get('image_size', [1920, 1080]))
        
        # Ground plane parameters
        self.ground_plane_normal = np.array(calibration_data.get('ground_plane_normal', [0, 0, 1]))
        self.ground_plane_point = np.array(calibration_data.get('ground_plane_point', [0, 0, 0]))
        
        # Scale factor (meters per pixel) as fallback
        self.pixel_to_meter_ratio = calibration_data.get('pixel_to_meter_ratio', None)
        
        # Validation
        self._validate_calibration()
        
    def _validate_calibration(self):
        """Validate calibration parameters"""
        if self.camera_matrix.shape != (3, 3):
            raise ValueError("Camera matrix must be 3x3")
            
        if self.homography_matrix.shape != (3, 3):
            raise ValueError("Homography matrix must be 3x3")
            
        if len(self.dist_coeffs) not in [4, 5, 8, 12, 14]:
            raise ValueError("Invalid number of distortion coefficients")
            
    @classmethod
    def from_file(cls, calibration_file: Union[str, Path]) -> 'CameraCalibration':
        """
        Load calibration from JSON file
        
        Args:
            calibration_file: Path to calibration file
            
        Returns:
            CameraCalibration instance
        """
        calibration_path = Path(calibration_file)
        
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
            
        with open(calibration_path, 'r') as f:
            calibration_data = json.load(f)
            
        return cls(calibration_data)
        
    def undistort_points(self, points: np.ndarray) -> np.ndarray:
        """
        Undistort image points using camera calibration
        
        Args:
            points: Array of image points (N, 2)
            
        Returns:
            Undistorted points (N, 2)
        """
        if len(points) == 0:
            return points
            
        points_reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        undistorted = cv2.undistortPoints(
            points_reshaped, 
            self.camera_matrix, 
            self.dist_coeffs,
            P=self.camera_matrix
        )
        
        return undistorted.reshape(-1, 2)
        
    def image_to_world(self, image_points: np.ndarray) -> np.ndarray:
        """
        Transform image coordinates to world coordinates using homography
        
        Args:
            image_points: Array of image points (N, 2)
            
        Returns:
            World coordinates (N, 2)
        """
        if len(image_points) == 0:
            return image_points
            
        # Undistort points first
        undistorted_points = self.undistort_points(image_points)
        
        # Convert to homogeneous coordinates
        homogeneous_points = np.column_stack([undistorted_points, np.ones(len(undistorted_points))])
        
        # Apply homography transformation
        world_homogeneous = (self.homography_matrix @ homogeneous_points.T).T
        
        # Convert back to Cartesian coordinates
        world_points = world_homogeneous[:, :2] / world_homogeneous[:, 2:3]
        
        return world_points
        
    def world_to_image(self, world_points: np.ndarray) -> np.ndarray:
        """
        Transform world coordinates to image coordinates using inverse homography
        
        Args:
            world_points: Array of world points (N, 2)
            
        Returns:
            Image coordinates (N, 2)
        """
        if len(world_points) == 0:
            return world_points
            
        # Use inverse homography
        inv_homography = np.linalg.inv(self.homography_matrix)
        
        # Convert to homogeneous coordinates
        homogeneous_points = np.column_stack([world_points, np.ones(len(world_points))])
        
        # Apply inverse homography
        image_homogeneous = (inv_homography @ homogeneous_points.T).T
        
        # Convert back to Cartesian coordinates
        image_points = image_homogeneous[:, :2] / image_homogeneous[:, 2:3]
        
        return image_points
        
    def simple_scale_transform(self, image_points: np.ndarray) -> np.ndarray:
        """
        Simple scaling transformation when homography is not available
        
        Args:
            image_points: Array of image points (N, 2)
            
        Returns:
            Scaled coordinates (N, 2)
        """
        if self.pixel_to_meter_ratio is None:
            raise ValueError("No pixel-to-meter ratio available for simple scaling")
            
        return image_points * self.pixel_to_meter_ratio


class VelocityCalculator:
    """
    Calculates real-world velocities from pixel coordinates.
    Uses camera calibration and temporal smoothing for accuracy.
    """
    
    def __init__(self, 
                 calibration: Union[str, Path, Dict, CameraCalibration],
                 fps: float = 30.0,
                 smoothing_window: int = 5,
                 min_track_length: int = 3,
                 max_track_age: float = 2.0,
                 outlier_threshold: float = 2.0,
                 velocity_smoothing: str = 'kalman'):  # 'moving_average', 'kalman', 'polynomial'
        """
        Initialize velocity calculator with camera calibration.
        
        Args:
            calibration: Camera calibration (file path, dict, or CameraCalibration object)
            fps: Video frame rate
            smoothing_window: Number of points for velocity smoothing
            min_track_length: Minimum track length for velocity calculation
            max_track_age: Maximum age of track in seconds before cleanup
            outlier_threshold: Standard deviations for outlier detection
            velocity_smoothing: Method for velocity smoothing
        """
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.min_track_length = min_track_length
        self.max_track_age = max_track_age
        self.outlier_threshold = outlier_threshold
        self.velocity_smoothing = velocity_smoothing
        
        # Load calibration
        if isinstance(calibration, (str, Path)):
            self.calibration = CameraCalibration.from_file(calibration)
        elif isinstance(calibration, dict):
            self.calibration = CameraCalibration(calibration)
        elif isinstance(calibration, CameraCalibration):
            self.calibration = calibration
        else:
            raise ValueError("Invalid calibration type")
            
        # Track histories
        self.track_histories: Dict[int, TrackHistory] = {}
        
        # Performance tracking
        self.calculation_times = []
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def update_track(self, 
                    track_id: int, 
                    bbox: List[float], 
                    frame_number: int,
                    timestamp: Optional[float] = None) -> Optional[VelocityMeasurement]:
        """
        Update track history and calculate velocity if possible.
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box [x1, y1, x2, y2]
            frame_number: Current frame number
            timestamp: Timestamp in seconds (auto-calculated if None)
            
        Returns:
            VelocityMeasurement if velocity can be calculated, None otherwise
        """
        start_time = time.time()
        
        try:
            # Calculate timestamp if not provided
            if timestamp is None:
                timestamp = frame_number / self.fps
                
            # Get bottom center of bounding box (vehicle ground contact point)
            image_point = self._get_ground_contact_point(bbox)
            
            # Transform to world coordinates
            try:
                world_point = self.calibration.image_to_world(np.array([image_point]))[0]
            except Exception as e:
                self.logger.warning(f"Homography transformation failed, using simple scaling: {e}")
                world_point = self.calibration.simple_scale_transform(np.array([image_point]))[0]
                
            # Initialize track history if new
            if track_id not in self.track_histories:
                self.track_histories[track_id] = TrackHistory(
                    positions_image=deque(maxlen=self.smoothing_window * 2),
                    positions_world=deque(maxlen=self.smoothing_window * 2),
                    timestamps=deque(maxlen=self.smoothing_window * 2),
                    frame_numbers=deque(maxlen=self.smoothing_window * 2),
                    velocities=deque(maxlen=self.smoothing_window),
                    last_update=timestamp
                )
                
            history = self.track_histories[track_id]
            
            # Add new data point
            history.positions_image.append(image_point)
            history.positions_world.append(world_point)
            history.timestamps.append(timestamp)
            history.frame_numbers.append(frame_number)
            history.last_update = timestamp
            
            # Calculate velocity if enough history
            velocity_measurement = None
            if len(history.positions_world) >= self.min_track_length:
                velocity_measurement = self._calculate_velocity(track_id, history, timestamp, frame_number)
                
            return velocity_measurement
            
        except Exception as e:
            self.logger.error(f"Error updating track {track_id}: {e}")
            return None
            
        finally:
            # Track performance
            calc_time = time.time() - start_time
            self.calculation_times.append(calc_time)
            if len(self.calculation_times) > 100:
                self.calculation_times = self.calculation_times[-100:]
                
    def _get_ground_contact_point(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Get ground contact point from bounding box.
        Uses bottom center as contact point for vehicles.
        """
        x1, y1, x2, y2 = bbox
        
        # Bottom center point
        center_x = (x1 + x2) / 2
        bottom_y = y2
        
        return (center_x, bottom_y)
        
    def _calculate_velocity(self, 
                          track_id: int, 
                          history: TrackHistory, 
                          timestamp: float,
                          frame_number: int) -> Optional[VelocityMeasurement]:
        """
        Calculate velocity using specified smoothing method
        """
        try:
            # Get recent positions and timestamps
            positions = np.array(list(history.positions_world))
            timestamps = np.array(list(history.timestamps))
            
            # Remove outliers
            positions_clean, timestamps_clean = self._remove_outliers(positions, timestamps)
            
            if len(positions_clean) < self.min_track_length:
                return None
                
            # Calculate velocity based on smoothing method
            if self.velocity_smoothing == 'moving_average':
                velocity_ms, direction_angle, confidence = self._calculate_moving_average_velocity(
                    positions_clean, timestamps_clean)
            elif self.velocity_smoothing == 'kalman':
                velocity_ms, direction_angle, confidence = self._calculate_kalman_velocity(
                    positions_clean, timestamps_clean)
            elif self.velocity_smoothing == 'polynomial':
                velocity_ms, direction_angle, confidence = self._calculate_polynomial_velocity(
                    positions_clean, timestamps_clean)
            else:
                raise ValueError(f"Unknown velocity smoothing method: {self.velocity_smoothing}")
                
            # Convert to km/h
            velocity_kmh = velocity_ms * 3.6
            
            # Store velocity in history
            history.velocities.append(velocity_ms)
            
            # Create measurement
            return VelocityMeasurement(
                track_id=track_id,
                velocity_kmh=velocity_kmh,
                velocity_ms=velocity_ms,
                world_position=tuple(positions_clean[-1]),
                direction_angle=direction_angle,
                confidence=confidence,
                timestamp=timestamp,
                frame_number=frame_number
            )
            
        except Exception as e:
            self.logger.error(f"Velocity calculation failed for track {track_id}: {e}")
            return None
            
    def _remove_outliers(self, positions: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove outlier positions using statistical filtering
        """
        if len(positions) < 3:
            return positions, timestamps
            
        # Calculate velocities between consecutive points
        velocities = []
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                dx = positions[i, 0] - positions[i-1, 0]
                dy = positions[i, 1] - positions[i-1, 1]
                velocity = np.sqrt(dx**2 + dy**2) / dt
                velocities.append(velocity)
            else:
                velocities.append(0)
                
        velocities = np.array(velocities)
        
        # Remove outliers based on velocity
        if len(velocities) > 3:
            velocity_mean = np.mean(velocities)
            velocity_std = np.std(velocities)
            threshold = velocity_mean + self.outlier_threshold * velocity_std
            
            # Keep points with reasonable velocities
            valid_indices = [0]  # Always keep first point
            for i, vel in enumerate(velocities):
                if vel <= threshold:
                    valid_indices.append(i + 1)
                    
            positions_clean = positions[valid_indices]
            timestamps_clean = timestamps[valid_indices]
        else:
            positions_clean = positions
            timestamps_clean = timestamps
            
        return positions_clean, timestamps_clean
        
    def _calculate_moving_average_velocity(self, positions: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate velocity using moving average of position derivatives
        """
        if len(positions) < 2:
            return 0.0, 0.0, 0.0
            
        # Calculate instantaneous velocities
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt > 0:
                vx = (positions[i, 0] - positions[i-1, 0]) / dt
                vy = (positions[i, 1] - positions[i-1, 1]) / dt
                velocities_x.append(vx)
                velocities_y.append(vy)
                
        if not velocities_x:
            return 0.0, 0.0, 0.0
            
        # Average velocities
        avg_vx = np.mean(velocities_x)
        avg_vy = np.mean(velocities_y)
        
        # Calculate speed and direction
        speed = np.sqrt(avg_vx**2 + avg_vy**2)
        direction_angle = np.degrees(np.arctan2(avg_vy, avg_vx))
        
        # Confidence based on velocity consistency
        velocity_magnitudes = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(velocities_x, velocities_y)]
        if len(velocity_magnitudes) > 1:
            velocity_std = np.std(velocity_magnitudes)
            velocity_mean = np.mean(velocity_magnitudes)
            confidence = max(0.0, 1.0 - (velocity_std / (velocity_mean + 1e-6)))
        else:
            confidence = 0.5
            
        return speed, direction_angle, confidence
        
    def _calculate_kalman_velocity(self, positions: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate velocity using Kalman filter for smooth estimation
        """
        if len(positions) < 2:
            return 0.0, 0.0, 0.0
            
        # Simple Kalman filter implementation
        # State: [x, y, vx, vy]
        state = np.array([positions[0, 0], positions[0, 1], 0.0, 0.0])
        
        # Process noise (motion model uncertainty)
        Q = np.eye(4) * 0.1
        Q[2:, 2:] *= 10  # Higher velocity uncertainty
        
        # Measurement noise (position measurement uncertainty)
        R = np.eye(2) * 0.5
        
        # State covariance
        P = np.eye(4) * 1.0
        
        velocities = []
        
        for i in range(1, len(positions)):
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                continue
                
            # State transition matrix
            F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            
            # Observation matrix
            H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
            
            # Predict
            state = F @ state
            P = F @ P @ F.T + Q
            
            # Update
            measurement = positions[i]
            innovation = measurement - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            
            state = state + K @ innovation
            P = (np.eye(4) - K @ H) @ P
            
            # Extract velocity
            vx, vy = state[2], state[3]
            velocities.append((vx, vy))
            
        if not velocities:
            return 0.0, 0.0, 0.0
            
        # Use final velocity estimate
        final_vx, final_vy = velocities[-1]
        speed = np.sqrt(final_vx**2 + final_vy**2)
        direction_angle = np.degrees(np.arctan2(final_vy, final_vx))
        
        # Confidence based on filter convergence
        confidence = min(1.0, len(velocities) / self.smoothing_window)
        
        return speed, direction_angle, confidence
        
    def _calculate_polynomial_velocity(self, positions: np.ndarray, timestamps: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate velocity using polynomial fitting and differentiation
        """
        if len(positions) < 3:
            return self._calculate_moving_average_velocity(positions, timestamps)
            
        # Normalize timestamps to start from 0
        t_norm = timestamps - timestamps[0]
        
        try:
            # Fit polynomial to trajectory (degree 2 for smooth velocity)
            degree = min(2, len(positions) - 1)
            
            # Fit x and y separately
            px_coeffs = np.polyfit(t_norm, positions[:, 0], degree)
            py_coeffs = np.polyfit(t_norm, positions[:, 1], degree)
            
            # Calculate velocity at final time (derivative of polynomial)
            t_final = t_norm[-1]
            
            # Derivative coefficients
            px_vel_coeffs = [i * px_coeffs[-(i+2)] for i in range(1, len(px_coeffs))]
            py_vel_coeffs = [i * py_coeffs[-(i+2)] for i in range(1, len(py_coeffs))]
            
            # Evaluate velocity at final time
            vx = sum(coeff * (t_final ** i) for i, coeff in enumerate(px_vel_coeffs))
            vy = sum(coeff * (t_final ** i) for i, coeff in enumerate(py_vel_coeffs))
            
            speed = np.sqrt(vx**2 + vy**2)
            direction_angle = np.degrees(np.arctan2(vy, vx))
            
            # Confidence based on fit quality (R-squared approximation)
            px_pred = np.polyval(px_coeffs, t_norm)
            py_pred = np.polyval(py_coeffs, t_norm)
            
            x_r2 = 1 - np.sum((positions[:, 0] - px_pred)**2) / np.sum((positions[:, 0] - np.mean(positions[:, 0]))**2)
            y_r2 = 1 - np.sum((positions[:, 1] - py_pred)**2) / np.sum((positions[:, 1] - np.mean(positions[:, 1]))**2)
            
            confidence = (x_r2 + y_r2) / 2
            confidence = max(0.0, min(1.0, confidence))
            
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to moving average if polynomial fitting fails
            return self._calculate_moving_average_velocity(positions, timestamps)
            
        return speed, direction_angle, confidence
        
    def cleanup_old_tracks(self, current_timestamp: float):
        """
        Remove old track histories to prevent memory leaks
        """
        tracks_to_remove = []
        
        for track_id, history in self.track_histories.items():
            if current_timestamp - history.last_update > self.max_track_age:
                tracks_to_remove.append(track_id)
                
        for track_id in tracks_to_remove:
            del self.track_histories[track_id]
            
        if tracks_to_remove:
            self.logger.debug(f"Cleaned up {len(tracks_to_remove)} old tracks")
            
    def get_track_history(self, track_id: int) -> Optional[TrackHistory]:
        """Get history for a specific track"""
        return self.track_histories.get(track_id)
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get calculation performance statistics"""
        if not self.calculation_times:
            return {}
            
        times = np.array(self.calculation_times)
        
        return {
            'mean_calculation_time': float(np.mean(times)),
            'median_calculation_time': float(np.median(times)),
            'min_calculation_time': float(np.min(times)),
            'max_calculation_time': float(np.max(times)),
            'std_calculation_time': float(np.std(times)),
            'total_calculations': len(self.calculation_times)
        }
        
    def reset_stats(self):
        """Reset performance statistics"""
        self.calculation_times = []
        
    def reset_all_tracks(self):
        """Reset all track histories"""
        self.track_histories.clear()


# Helper functions for creating calibrations and calculators
def create_simple_calibration(image_size: Tuple[int, int], 
                            pixel_to_meter_ratio: float) -> CameraCalibration:
    """
    Create simple calibration with just pixel-to-meter scaling
    
    Args:
        image_size: Image dimensions (width, height)
        pixel_to_meter_ratio: Meters per pixel scaling factor
        
    Returns:
        CameraCalibration instance
    """
    calibration_data = {
        'camera_matrix': np.eye(3).tolist(),
        'dist_coeffs': [0, 0, 0, 0, 0],
        'homography_matrix': np.eye(3).tolist(),
        'image_size': list(image_size),
        'pixel_to_meter_ratio': pixel_to_meter_ratio
    }
    
    return CameraCalibration(calibration_data)


def create_velocity_calculator(calibration_file: Union[str, Path], **kwargs) -> VelocityCalculator:
    """
    Create velocity calculator from calibration file
    
    Args:
        calibration_file: Path to calibration file
        **kwargs: Additional arguments for VelocityCalculator
        
    Returns:
        Initialized VelocityCalculator
    """
    return VelocityCalculator(calibration_file, **kwargs)