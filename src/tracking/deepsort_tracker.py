"""
DeepSORT implementation optimized for vehicle tracking.
Maintains consistent IDs across frames for velocity calculation.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import time
from pathlib import Path

# Try to import deep_sort components, provide fallback implementations
try:
    from deep_sort.detection import Detection as DSDetection
    from deep_sort.tracker import Tracker
    from deep_sort.nn_matching import NearestNeighborDistanceMetric
    DEEP_SORT_AVAILABLE = True
except ImportError:
    DEEP_SORT_AVAILABLE = False


@dataclass
class Track:
    """
    Represents a tracked object with consistent ID
    """
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    age: int
    hits: int
    time_since_update: int
    state: str  # 'tentative', 'confirmed', 'deleted'
    features: Optional[np.ndarray] = None
    
    def to_tlbr(self) -> np.ndarray:
        """Convert to top-left bottom-right format"""
        return np.array(self.bbox)
        
    def to_tlwh(self) -> np.ndarray:
        """Convert to top-left width-height format"""
        x1, y1, x2, y2 = self.bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])
        
    def get_center(self) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
        
    def get_area(self) -> float:
        """Get area of bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class SimpleFeatureExtractor(nn.Module):
    """
    Lightweight feature extractor for vehicle re-identification.
    Can be replaced with more sophisticated models.
    """
    
    def __init__(self, input_size: Tuple[int, int] = (128, 256), feature_dim: int = 128):
        super().__init__()
        self.input_size = input_size
        self.feature_dim = feature_dim
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature projection
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images"""
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # L2 normalize features
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x


class KalmanFilter:
    """
    Simple Kalman filter for tracking bounding box motion.
    Simplified implementation for when deep_sort is not available.
    """
    
    def __init__(self):
        self.ndim = 4
        self.dt = 1.0
        
        # State transition matrix
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
            
        # Observation matrix
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Motion and observation uncertainty
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create initial state and covariance"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
        
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict next state"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
            
        return mean, covariance
        
    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Update state with measurement"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean, covariance = self._update(mean, covariance, measurement, innovation_cov)
        return mean, covariance
        
    def _update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray, innovation_cov: np.ndarray):
        """Internal update method"""
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov + innovation_cov, lower=True)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T)
        innovation = measurement - projected_mean
        
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
            
        return new_mean, new_covariance


class SimpleTracker:
    """
    Simplified tracking implementation when DeepSORT is not available.
    Uses IoU matching and simple motion prediction.
    """
    
    def __init__(self, 
                 max_age: int = 30,
                 min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
    def update(self, detections: List[Dict]) -> List[Track]:
        """Update tracks with new detections"""
        self.frame_count += 1
        
        # Convert detections to proper format
        detection_boxes = []
        for det in detections:
            bbox = det['bbox']
            # Ensure bbox is in [x1, y1, x2, y2] format
            if len(bbox) == 4:
                detection_boxes.append(bbox)
                
        # Predict current state for existing tracks
        for track in self.tracks:
            track.time_since_update += 1
            
        # Associate detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detection_boxes, self.tracks)
            
        # Update matched tracks
        for m in matched:
            det_idx, trk_idx = m
            self.tracks[trk_idx] = self._update_track(
                self.tracks[trk_idx], detections[det_idx])
                
        # Create new tracks for unmatched detections
        for i in unmatched_dets:
            self.tracks.append(self._create_track(detections[i]))
            
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Return confirmed tracks
        return [t for t in self.tracks if t.state == 'confirmed']
        
    def _associate_detections_to_tracks(self, detections: List, tracks: List[Track]):
        """Associate detections to tracks using IoU"""
        if len(tracks) == 0:
            return [], list(range(len(detections))), []
            
        if len(detections) == 0:
            return [], [], list(range(len(tracks)))
            
        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for d, det_bbox in enumerate(detections):
            for t, track in enumerate(tracks):
                iou_matrix[d, t] = self._compute_iou(det_bbox, track.bbox)
                
        # Hungarian assignment (simplified greedy approach)
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(tracks)))
        
        # Greedy matching
        while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            # Find best match
            best_iou = 0
            best_det, best_trk = -1, -1
            
            for d in unmatched_detections:
                for t in unmatched_tracks:
                    if iou_matrix[d, t] > best_iou:
                        best_iou = iou_matrix[d, t]
                        best_det, best_trk = d, t
                        
            if best_iou > self.iou_threshold:
                matched_indices.append([best_det, best_trk])
                unmatched_detections.remove(best_det)
                unmatched_tracks.remove(best_trk)
            else:
                break
                
        return matched_indices, unmatched_detections, unmatched_tracks
        
    def _compute_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
            
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _create_track(self, detection: Dict) -> Track:
        """Create new track from detection"""
        track = Track(
            track_id=self.next_id,
            bbox=detection['bbox'],
            confidence=detection['confidence'],
            class_id=detection.get('class_id', 0),
            class_name=detection.get('class_name', 'unknown'),
            age=1,
            hits=1,
            time_since_update=0,
            state='tentative'
        )
        
        self.next_id += 1
        return track
        
    def _update_track(self, track: Track, detection: Dict) -> Track:
        """Update existing track with new detection"""
        track.bbox = detection['bbox']
        track.confidence = detection['confidence']
        track.age += 1
        track.hits += 1
        track.time_since_update = 0
        
        # Confirm track if it has enough hits
        if track.hits >= self.min_hits:
            track.state = 'confirmed'
            
        return track


class DeepSORTTracker:
    """
    DeepSORT implementation optimized for vehicle tracking.
    Maintains consistent IDs across frames for velocity calculation.
    """
    
    def __init__(self,
                 feature_extractor_path: Optional[str] = None,
                 max_cosine_distance: float = 0.2,
                 nn_budget: int = 100,
                 max_age: int = 30,
                 n_init: int = 3,
                 max_iou_distance: float = 0.7,
                 device: str = 'auto'):
        """
        Initialize DeepSORT with parameters tuned for vehicle tracking.
        
        Args:
            feature_extractor_path: Path to feature extractor model
            max_cosine_distance: Maximum cosine distance for appearance matching
            nn_budget: Maximum number of features per class to store
            max_age: Maximum number of frames to keep track without detection
            n_init: Number of consecutive detections before track is confirmed
            max_iou_distance: Maximum IoU distance for spatial matching
            device: Device for feature extraction ('auto', 'cpu', 'cuda')
        """
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize tracking components
        if DEEP_SORT_AVAILABLE:
            self._init_deep_sort()
        else:
            self.logger.warning("DeepSORT not available, using simplified tracker")
            self.tracker = SimpleTracker(
                max_age=max_age,
                min_hits=n_init,
                iou_threshold=1.0 - max_iou_distance
            )
            
        # Load feature extractor
        self.feature_extractor = self._load_feature_extractor(feature_extractor_path)
        
        # Track performance
        self.tracking_times = []
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            
        return torch.device(device)
        
    def _init_deep_sort(self):
        """Initialize DeepSORT components"""
        # Distance metric for appearance features
        metric = NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget
        )
        
        # Initialize tracker
        self.tracker = Tracker(
            metric,
            max_iou_distance=self.max_iou_distance,
            max_age=self.max_age,
            n_init=self.n_init
        )
        
        self.logger.info("Initialized DeepSORT tracker")
        
    def _load_feature_extractor(self, model_path: Optional[str]) -> nn.Module:
        """
        Load feature extractor model for appearance matching
        """
        if model_path and Path(model_path).exists():
            try:
                # Load custom model
                model = torch.load(model_path, map_location=self.device)
                self.logger.info(f"Loaded custom feature extractor from {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load custom model: {e}, using default")
                model = SimpleFeatureExtractor()
        else:
            # Use simple feature extractor
            model = SimpleFeatureExtractor()
            self.logger.info("Using default simple feature extractor")
            
        model.to(self.device)
        model.eval()
        
        return model
        
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame for feature extraction
            
        Returns:
            List of active tracks with consistent IDs
        """
        start_time = time.time()
        
        try:
            if DEEP_SORT_AVAILABLE and hasattr(self.tracker, 'predict'):
                # DeepSORT update
                return self._update_deep_sort(detections, frame)
            else:
                # Simple tracker update
                return self._update_simple(detections, frame)
                
        finally:
            # Track performance
            self.tracking_times.append(time.time() - start_time)
            if len(self.tracking_times) > 100:
                self.tracking_times = self.tracking_times[-100:]
                
    def _update_deep_sort(self, detections: List[Dict], frame: np.ndarray) -> List[Track]:
        """Update using full DeepSORT implementation"""
        # Extract appearance features
        features = self._extract_features(detections, frame)
        
        # Convert to DeepSORT format
        ds_detections = []
        for det, feature in zip(detections, features):
            bbox = det['bbox']
            confidence = det['confidence']
            
            # Convert to [x, y, w, h] format for DeepSORT
            bbox_xywh = self._xyxy_to_xywh(bbox)
            
            ds_detections.append(
                DSDetection(bbox_xywh, confidence, feature)
            )
            
        # Update tracker
        self.tracker.predict()
        self.tracker.update(ds_detections)
        
        # Convert back to our Track format
        tracks = []
        for track in self.tracker.tracks:
            if track.is_confirmed():
                bbox_xyxy = self._xywh_to_xyxy(track.to_tlwh())
                
                tracks.append(Track(
                    track_id=track.track_id,
                    bbox=bbox_xyxy.tolist(),
                    confidence=0.0,  # Not available in DeepSORT track
                    class_id=0,  # Would need to store this separately
                    class_name='vehicle',
                    age=track.age,
                    hits=track.hits,
                    time_since_update=track.time_since_update,
                    state='confirmed'
                ))
                
        return tracks
        
    def _update_simple(self, detections: List[Dict], frame: np.ndarray) -> List[Track]:
        """Update using simplified tracker"""
        return self.tracker.update(detections)
        
    def _extract_features(self, detections: List[Dict], frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract appearance features for vehicle re-identification
        """
        if not detections:
            return []
            
        features = []
        crops = []
        
        # Extract crops for all detections
        for det in detections:
            crop = self._extract_crop(det['bbox'], frame)
            if crop is not None:
                crops.append(crop)
            else:
                # Use dummy feature if crop extraction fails
                features.append(np.zeros(128, dtype=np.float32))
                
        if not crops:
            return features
            
        # Batch process crops
        try:
            with torch.no_grad():
                # Prepare batch
                batch_tensor = torch.stack([
                    self._prepare_crop_tensor(crop) for crop in crops
                ])
                batch_tensor = batch_tensor.to(self.device)
                
                # Extract features
                batch_features = self.feature_extractor(batch_tensor)
                batch_features = batch_features.cpu().numpy()
                
                features.extend(batch_features)
                
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            # Return dummy features
            features = [np.zeros(128, dtype=np.float32) for _ in detections]
            
        return features
        
    def _extract_crop(self, bbox: List[float], frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract and preprocess crop from frame"""
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Ensure coordinates are within frame bounds
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            
            # Extract crop
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return None
                
            # Resize to standard size
            crop_resized = cv2.resize(crop, (128, 256))
            
            return crop_resized
            
        except Exception as e:
            self.logger.warning(f"Failed to extract crop: {e}")
            return None
            
    def _prepare_crop_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """Prepare crop for feature extraction"""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Standard normalization (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to tensor and rearrange dimensions
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        
        return tensor
        
    def _xyxy_to_xywh(self, bbox: List[float]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, w, h]"""
        x1, y1, x2, y2 = bbox
        return np.array([x1, y1, x2 - x1, y2 - y1])
        
    def _xywh_to_xyxy(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x, y, w, h] to [x1, y1, x2, y2]"""
        x, y, w, h = bbox
        return np.array([x, y, x + w, y + h])
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get tracking performance statistics"""
        if not self.tracking_times:
            return {}
            
        times = np.array(self.tracking_times)
        
        return {
            'mean_tracking_time': float(np.mean(times)),
            'median_tracking_time': float(np.median(times)),
            'min_tracking_time': float(np.min(times)),
            'max_tracking_time': float(np.max(times)),
            'std_tracking_time': float(np.std(times)),
            'total_updates': len(self.tracking_times)
        }
        
    def reset_stats(self):
        """Reset performance statistics"""
        self.tracking_times = []
        
    def reset_tracker(self):
        """Reset tracker state (useful for new video sequences)"""
        if DEEP_SORT_AVAILABLE and hasattr(self.tracker, 'tracks'):
            self.tracker.tracks = []
        elif hasattr(self.tracker, 'tracks'):
            self.tracker.tracks = []
            self.tracker.next_id = 1
            self.tracker.frame_count = 0


# Helper function for easy tracker creation
def create_tracker(config: Optional[Dict[str, Any]] = None) -> DeepSORTTracker:
    """
    Create DeepSORT tracker with configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized DeepSORTTracker
    """
    if config is None:
        config = {}
        
    return DeepSORTTracker(**config)