"""
Unit tests for DeepSORT tracking module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from tracking.deepsort_tracker import (
        Track, DeepSORTTracker, SimpleTracker, SimpleFeatureExtractor, create_tracker
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestTrack:
    """Test Track dataclass functionality"""
    
    def test_track_creation(self):
        """Test creating Track objects"""
        track = Track(
            track_id=1,
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_id=2,
            class_name='car',
            age=5,
            hits=3,
            time_since_update=0,
            state='confirmed'
        )
        
        assert track.track_id == 1
        assert track.bbox == [100, 100, 200, 200]
        assert track.confidence == 0.9
        assert track.state == 'confirmed'
        
    def test_track_conversions(self):
        """Test bounding box format conversions"""
        track = Track(
            track_id=1,
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_id=2,
            class_name='car',
            age=1,
            hits=1,
            time_since_update=0,
            state='confirmed'
        )
        
        # Test to_tlbr
        tlbr = track.to_tlbr()
        np.testing.assert_array_equal(tlbr, [100, 100, 200, 200])
        
        # Test to_tlwh
        tlwh = track.to_tlwh()
        np.testing.assert_array_equal(tlwh, [100, 100, 100, 100])
        
        # Test get_center
        center = track.get_center()
        assert center == (150.0, 150.0)
        
        # Test get_area
        area = track.get_area()
        assert area == 10000.0  # 100 * 100


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestSimpleFeatureExtractor:
    """Test SimpleFeatureExtractor"""
    
    def test_feature_extractor_creation(self):
        """Test creating feature extractor"""
        extractor = SimpleFeatureExtractor()
        
        assert extractor.input_size == (128, 256)
        assert extractor.feature_dim == 128
        
    def test_custom_feature_extractor(self):
        """Test feature extractor with custom parameters"""
        extractor = SimpleFeatureExtractor(
            input_size=(64, 128),
            feature_dim=256
        )
        
        assert extractor.input_size == (64, 128)
        assert extractor.feature_dim == 256


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestSimpleTracker:
    """Test SimpleTracker functionality"""
    
    def test_simple_tracker_creation(self):
        """Test creating simple tracker"""
        tracker = SimpleTracker(
            max_age=20,
            min_hits=2,
            iou_threshold=0.5
        )
        
        assert tracker.max_age == 20
        assert tracker.min_hits == 2
        assert tracker.iou_threshold == 0.5
        assert len(tracker.tracks) == 0
        assert tracker.next_id == 1
        
    def test_iou_computation(self):
        """Test IoU computation"""
        tracker = SimpleTracker()
        
        # Identical boxes
        iou = tracker._compute_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert abs(iou - 1.0) < 1e-6
        
        # No overlap
        iou = tracker._compute_iou([0, 0, 50, 50], [100, 100, 150, 150])
        assert abs(iou - 0.0) < 1e-6
        
        # Partial overlap
        iou = tracker._compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        expected = 2500 / 17500  # intersection / union
        assert abs(iou - expected) < 1e-6
        
    def test_track_creation(self):
        """Test track creation from detection"""
        tracker = SimpleTracker()
        
        detection = {
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 2,
            'class_name': 'car'
        }
        
        track = tracker._create_track(detection)
        
        assert track.track_id == 1
        assert track.bbox == [100, 100, 200, 200]
        assert track.confidence == 0.9
        assert track.class_id == 2
        assert track.state == 'tentative'
        assert track.hits == 1
        
    def test_simple_tracking_update(self):
        """Test updating tracker with detections"""
        tracker = SimpleTracker(min_hits=1)  # Confirm immediately for testing
        
        # First frame - new detection
        detections1 = [{
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 2,
            'class_name': 'car'
        }]
        
        tracks1 = tracker.update(detections1)
        assert len(tracks1) == 1
        assert tracks1[0].track_id == 1
        assert tracks1[0].state == 'confirmed'
        
        # Second frame - same detection moved slightly
        detections2 = [{
            'bbox': [105, 105, 205, 205],
            'confidence': 0.85,
            'class_id': 2,
            'class_name': 'car'
        }]
        
        tracks2 = tracker.update(detections2)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == 1  # Same ID
        assert tracks2[0].bbox == [105, 105, 205, 205]  # Updated position
        
    def test_track_aging(self):
        """Test track aging and deletion"""
        tracker = SimpleTracker(max_age=2, min_hits=1)
        
        # Create track
        detections = [{
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 2,
            'class_name': 'car'
        }]
        
        tracks = tracker.update(detections)
        assert len(tracks) == 1
        
        # Update without detections - track should age
        tracks = tracker.update([])
        assert len(tracks) == 1  # Still alive
        
        tracks = tracker.update([])
        assert len(tracks) == 1  # Still alive (age = 2)
        
        tracks = tracker.update([])
        assert len(tracks) == 0  # Now dead (age > max_age)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestDeepSORTTracker:
    """Test DeepSORTTracker functionality"""
    
    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = DeepSORTTracker(
            max_age=25,
            n_init=2,
            device='cpu'
        )
        
        assert tracker.max_age == 25
        assert tracker.n_init == 2
        assert tracker.device.type == 'cpu'
        assert tracker.feature_extractor is not None
        
    def test_device_setup(self):
        """Test device setup"""
        # CPU device
        tracker = DeepSORTTracker(device='cpu')
        assert tracker.device.type == 'cpu'
        
        # Auto device selection
        with patch('torch.cuda.is_available', return_value=False):
            tracker = DeepSORTTracker(device='auto')
            assert tracker.device.type == 'cpu'
            
    def test_bbox_conversions(self):
        """Test bounding box format conversions"""
        tracker = DeepSORTTracker()
        
        # Test xyxy to xywh
        xyxy = [100, 100, 200, 200]
        xywh = tracker._xyxy_to_xywh(xyxy)
        np.testing.assert_array_equal(xywh, [100, 100, 100, 100])
        
        # Test xywh to xyxy
        xywh_arr = np.array([100, 100, 100, 100])
        xyxy_converted = tracker._xywh_to_xyxy(xywh_arr)
        np.testing.assert_array_equal(xyxy_converted, [100, 100, 200, 200])
        
    def test_crop_extraction(self):
        """Test crop extraction from frame"""
        tracker = DeepSORTTracker()
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test valid bbox
        bbox = [100, 100, 200, 200]
        crop = tracker._extract_crop(bbox, frame)
        
        assert crop is not None
        assert crop.shape == (256, 128, 3)  # Resized to standard size
        
        # Test invalid bbox (out of bounds)
        bbox_invalid = [600, 400, 700, 500]  # Outside frame
        crop_invalid = tracker._extract_crop(bbox_invalid, frame)
        
        # Should handle gracefully
        assert crop_invalid is None or crop_invalid.shape == (256, 128, 3)
        
    def test_update_with_detections(self):
        """Test updating tracker with detections"""
        tracker = DeepSORTTracker(device='cpu')
        
        # Create test frame and detections
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = [{
            'bbox': [100, 100, 200, 200],
            'confidence': 0.9,
            'class_id': 2,
            'class_name': 'car'
        }]
        
        # Update tracker
        tracks = tracker.update(detections, frame)
        
        # Should return tracks (exact behavior depends on implementation)
        assert isinstance(tracks, list)
        
    def test_performance_stats(self):
        """Test performance statistics"""
        tracker = DeepSORTTracker()
        
        # No stats initially
        stats = tracker.get_performance_stats()
        assert stats == {}
        
        # Add some tracking times
        tracker.tracking_times = [0.1, 0.2, 0.15]
        
        stats = tracker.get_performance_stats()
        
        assert 'mean_tracking_time' in stats
        assert 'total_updates' in stats
        assert stats['total_updates'] == 3
        
    def test_reset_functions(self):
        """Test reset functionality"""
        tracker = DeepSORTTracker()
        
        # Add some data
        tracker.tracking_times = [0.1, 0.2, 0.15]
        
        # Reset stats
        tracker.reset_stats()
        assert tracker.tracking_times == []
        
        # Reset tracker should not crash
        tracker.reset_tracker()


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestCreateTracker:
    """Test tracker creation helper function"""
    
    def test_create_tracker_default(self):
        """Test creating tracker with default config"""
        tracker = create_tracker()
        
        assert isinstance(tracker, DeepSORTTracker)
        
    def test_create_tracker_with_config(self):
        """Test creating tracker with custom config"""
        config = {
            'max_age': 40,
            'n_init': 5,
            'device': 'cpu'
        }
        
        tracker = create_tracker(config)
        
        assert isinstance(tracker, DeepSORTTracker)
        assert tracker.max_age == 40
        assert tracker.n_init == 5


def test_module_structure():
    """Test module structure without heavy dependencies"""
    module_path = Path(__file__).parent.parent.parent / "src" / "tracking" / "deepsort_tracker.py"
    assert module_path.exists()
    
    # Check that the file contains expected class definitions
    content = module_path.read_text()
    assert "class DeepSORTTracker" in content
    assert "class Track" in content
    assert "class SimpleTracker" in content
    assert "def create_tracker" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])