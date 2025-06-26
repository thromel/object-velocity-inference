"""
Unit tests for YOLOv9 detector module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from models.yolo_detector import YOLOv9Detector, Detection, create_detector
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestDetection:
    """Test Detection dataclass"""
    
    def test_detection_creation(self):
        """Test creating Detection objects"""
        detection = Detection(
            bbox=[100, 100, 200, 200],
            confidence=0.9,
            class_id=2,
            class_name='car'
        )
        
        assert detection.bbox == [100, 100, 200, 200]
        assert detection.confidence == 0.9
        assert detection.class_id == 2
        assert detection.class_name == 'car'


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestYOLOv9Detector:
    """Test YOLOv9Detector functionality"""
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = YOLOv9Detector(
            model_path="test_model.pt",
            device="cpu",
            conf_threshold=0.6,
            iou_threshold=0.4
        )
        
        assert detector.conf_threshold == 0.6
        assert detector.iou_threshold == 0.4
        assert detector.device.type == "cpu"
        assert not detector.model_loaded
        assert detector.target_classes == [2, 3, 5, 7]  # Default vehicle classes
        
    def test_device_setup_auto(self):
        """Test automatic device selection"""
        with patch('torch.cuda.is_available', return_value=True):
            detector = YOLOv9Detector("test_model.pt", device="auto")
            assert "cuda" in str(detector.device)
            
        with patch('torch.cuda.is_available', return_value=False):
            detector = YOLOv9Detector("test_model.pt", device="auto")
            assert detector.device.type == "cpu"
            assert not detector.half_precision  # Should disable FP16 on CPU
            
    def test_vehicle_classes(self):
        """Test vehicle class mapping"""
        detector = YOLOv9Detector("test_model.pt")
        
        expected_classes = {
            2: 'car',
            3: 'motorcycle',
            5: 'bus', 
            7: 'truck'
        }
        
        assert detector.VEHICLE_CLASSES == expected_classes
        
    def test_custom_target_classes(self):
        """Test custom target classes"""
        custom_classes = [2, 5]  # Only cars and buses
        detector = YOLOv9Detector("test_model.pt", target_classes=custom_classes)
        
        assert detector.target_classes == custom_classes
        
    def test_letterbox_resize(self):
        """Test letterbox resizing functionality"""
        detector = YOLOv9Detector("test_model.pt")
        
        # Test with square image
        square_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = detector._letterbox_resize(square_img, (640, 640))
        
        assert resized.shape == (640, 640, 3)
        
        # Test with rectangular image
        rect_img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized = detector._letterbox_resize(rect_img, (640, 640))
        
        assert resized.shape == (640, 640, 3)
        
    def test_scale_coordinates(self):
        """Test coordinate scaling"""
        detector = YOLOv9Detector("test_model.pt", img_size=640)
        
        # Test box in 640x640 model input scaled to 1280x720 original
        model_box = np.array([100, 100, 200, 200])  # Box in model coordinates
        original_shape = (720, 1280, 3)  # Original image shape
        
        scaled_box = detector._scale_coordinates(model_box, original_shape)
        
        # Coordinates should be scaled and within bounds
        assert 0 <= scaled_box[0] <= 1280  # x1
        assert 0 <= scaled_box[1] <= 720   # y1
        assert 0 <= scaled_box[2] <= 1280  # x2
        assert 0 <= scaled_box[3] <= 720   # y3
        
    @patch('torch.load')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_model_torch(self, mock_exists, mock_torch_load):
        """Test model loading with torch.load"""
        # Mock model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_torch_load.return_value = mock_model
        
        detector = YOLOv9Detector("test_model.pt", device="cpu")
        
        # Mock ultralytics import failure to use torch.load
        with patch('builtins.__import__', side_effect=ImportError("No ultralytics")):
            result = detector.load_model()
            
        assert result
        assert detector.model_loaded
        assert detector.model == mock_model
        mock_model.eval.assert_called_once()
        
    def test_load_model_file_not_found(self):
        """Test model loading with missing file"""
        detector = YOLOv9Detector("nonexistent_model.pt")
        
        result = detector.load_model()
        
        assert not result
        assert not detector.model_loaded
        
    def test_preprocess_frame(self):
        """Test frame preprocessing"""
        detector = YOLOv9Detector("test_model.pt", img_size=640)
        
        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Mock ultralytics model
        detector.model = Mock()
        detector.model.predict = Mock()
        
        processed = detector._preprocess_frame(frame)
        
        # Should return numpy array for ultralytics
        assert isinstance(processed, np.ndarray)
        assert processed.shape == (640, 640, 3)
        
    def test_detect_no_model(self):
        """Test detection without loaded model"""
        detector = YOLOv9Detector("test_model.pt")
        
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert detections == []
        
    @patch('models.yolo_detector.YOLOv9Detector.load_model', return_value=True)
    def test_detect_with_mock_model(self, mock_load):
        """Test detection with mocked model"""
        detector = YOLOv9Detector("test_model.pt")
        detector.model_loaded = True
        
        # Mock ultralytics model and results
        mock_model = Mock()
        mock_boxes = Mock()
        mock_boxes.xyxy = [np.array([100, 100, 200, 200])]  # One detection
        mock_boxes.conf = [0.9]
        mock_boxes.cls = [2]  # Car class
        
        mock_result = Mock()
        mock_result.boxes = mock_boxes
        
        mock_model.predict.return_value = [mock_result]
        detector.model = mock_model
        
        # Test detection
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        assert len(detections) == 1
        assert detections[0].class_id == 2
        assert detections[0].class_name == 'car'
        assert detections[0].confidence == 0.9
        
    def test_detect_batch_empty(self):
        """Test batch detection with empty input"""
        detector = YOLOv9Detector("test_model.pt")
        
        results = detector.detect_batch([])
        assert results == []
        
    def test_performance_stats_empty(self):
        """Test performance stats with no inferences"""
        detector = YOLOv9Detector("test_model.pt")
        
        stats = detector.get_performance_stats()
        assert stats == {}
        
    def test_performance_stats_with_data(self):
        """Test performance stats with inference times"""
        detector = YOLOv9Detector("test_model.pt")
        
        # Add some inference times
        detector.inference_times = [0.1, 0.2, 0.15, 0.18]
        
        stats = detector.get_performance_stats()
        
        assert 'mean_inference_time' in stats
        assert 'fps_estimate' in stats
        assert 'total_inferences' in stats
        assert stats['total_inferences'] == 4
        assert stats['fps_estimate'] > 0
        
    def test_reset_stats(self):
        """Test resetting performance statistics"""
        detector = YOLOv9Detector("test_model.pt")
        detector.inference_times = [0.1, 0.2, 0.15]
        
        detector.reset_stats()
        
        assert detector.inference_times == []


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestCreateDetector:
    """Test detector creation helper function"""
    
    @patch('models.yolo_detector.YOLOv9Detector.load_model', return_value=True)
    def test_create_detector(self, mock_load):
        """Test create_detector helper function"""
        detector = create_detector(
            "test_model.pt",
            device="cpu",
            conf_threshold=0.7
        )
        
        assert isinstance(detector, YOLOv9Detector)
        assert detector.conf_threshold == 0.7
        assert detector.device.type == "cpu"
        mock_load.assert_called_once()


def test_module_structure():
    """Test module structure without heavy dependencies"""
    # This test can run even without torch/cv2
    module_path = Path(__file__).parent.parent.parent / "src" / "models" / "yolo_detector.py"
    assert module_path.exists()
    
    # Check that the file contains expected class definitions
    content = module_path.read_text()
    assert "class YOLOv9Detector" in content
    assert "class Detection" in content
    assert "def create_detector" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])