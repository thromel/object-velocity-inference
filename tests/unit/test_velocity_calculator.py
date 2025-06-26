"""
Unit tests for velocity calculator module.
"""

import pytest
import numpy as np
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from velocity.velocity_calculator import (
        VelocityCalculator, CameraCalibration, VelocityMeasurement, TrackHistory,
        create_simple_calibration, create_velocity_calculator
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)
    # Create dummy classes for type hints when imports fail
    class CameraCalibration: pass
    class VelocityCalculator: pass
    class VelocityMeasurement: pass


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestVelocityMeasurement:
    """Test VelocityMeasurement dataclass"""
    
    def test_velocity_measurement_creation(self):
        """Test creating VelocityMeasurement objects"""
        measurement = VelocityMeasurement(
            track_id=1,
            velocity_kmh=60.0,
            velocity_ms=16.67,
            world_position=(10.0, 5.0),
            direction_angle=45.0,
            confidence=0.9,
            timestamp=1.5,
            frame_number=45
        )
        
        assert measurement.track_id == 1
        assert measurement.velocity_kmh == 60.0
        assert measurement.velocity_ms == 16.67
        assert measurement.world_position == (10.0, 5.0)
        assert measurement.direction_angle == 45.0
        assert measurement.confidence == 0.9
        assert measurement.timestamp == 1.5
        assert measurement.frame_number == 45


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestCameraCalibration:
    """Test CameraCalibration functionality"""
    
    def create_test_calibration_data(self) -> dict:
        """Create test calibration data"""
        return {
            'camera_matrix': np.eye(3).tolist(),
            'dist_coeffs': [0.1, -0.2, 0.0, 0.0, 0.0],
            'homography_matrix': np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]).tolist(),
            'image_size': [1920, 1080],
            'pixel_to_meter_ratio': 0.05
        }
        
    def test_calibration_creation(self):
        """Test creating CameraCalibration from data"""
        cal_data = self.create_test_calibration_data()
        calibration = CameraCalibration(cal_data)
        
        assert calibration.camera_matrix.shape == (3, 3)
        assert calibration.homography_matrix.shape == (3, 3)
        assert len(calibration.dist_coeffs) == 5
        assert calibration.image_size == (1920, 1080)
        assert calibration.pixel_to_meter_ratio == 0.05
        
    def test_calibration_from_file(self):
        """Test loading calibration from file"""
        cal_data = self.create_test_calibration_data()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cal_data, f)
            temp_path = f.name
            
        try:
            calibration = CameraCalibration.from_file(temp_path)
            assert calibration.image_size == (1920, 1080)
        finally:
            Path(temp_path).unlink()
            
    def test_calibration_validation(self):
        """Test calibration validation"""
        # Invalid camera matrix
        with pytest.raises(ValueError, match="Camera matrix must be 3x3"):
            CameraCalibration({
                'camera_matrix': [[1, 0], [0, 1]],  # Wrong size
                'dist_coeffs': [0, 0, 0, 0, 0],
                'homography_matrix': np.eye(3).tolist()
            })
            
        # Invalid distortion coefficients
        with pytest.raises(ValueError, match="Invalid number of distortion coefficients"):
            CameraCalibration({
                'camera_matrix': np.eye(3).tolist(),
                'dist_coeffs': [0, 0, 0],  # Wrong number
                'homography_matrix': np.eye(3).tolist()
            })
            
    def test_image_to_world_transform(self):
        """Test image to world coordinate transformation"""
        cal_data = self.create_test_calibration_data()
        calibration = CameraCalibration(cal_data)
        
        # Test with identity homography (should preserve coordinates)
        image_points = np.array([[100, 200], [300, 400]])
        world_points = calibration.image_to_world(image_points)
        
        assert world_points.shape == (2, 2)
        # With identity matrix and no distortion, should be close to original
        np.testing.assert_allclose(world_points, image_points, rtol=0.1)
        
    def test_world_to_image_transform(self):
        """Test world to image coordinate transformation"""
        cal_data = self.create_test_calibration_data()
        calibration = CameraCalibration(cal_data)
        
        # Test roundtrip transformation
        original_points = np.array([[100, 200], [300, 400]])
        world_points = calibration.image_to_world(original_points)
        image_points = calibration.world_to_image(world_points)
        
        # Should be close to original after roundtrip
        np.testing.assert_allclose(image_points, original_points, rtol=0.1)
        
    def test_simple_scale_transform(self):
        """Test simple scaling transformation"""
        cal_data = self.create_test_calibration_data()
        calibration = CameraCalibration(cal_data)
        
        image_points = np.array([[100, 200]])
        scaled_points = calibration.simple_scale_transform(image_points)
        
        expected = image_points * 0.05  # pixel_to_meter_ratio
        np.testing.assert_array_equal(scaled_points, expected)
        
    def test_empty_points(self):
        """Test handling of empty point arrays"""
        cal_data = self.create_test_calibration_data()
        calibration = CameraCalibration(cal_data)
        
        empty_points = np.array([]).reshape(0, 2)
        
        # Should handle empty arrays gracefully
        result = calibration.image_to_world(empty_points)
        assert result.shape == (0, 2)
        
        result = calibration.world_to_image(empty_points)
        assert result.shape == (0, 2)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestVelocityCalculator:
    """Test VelocityCalculator functionality"""
    
    def create_test_calibration(self) -> CameraCalibration:
        """Create test calibration"""
        cal_data = {
            'camera_matrix': np.eye(3).tolist(),
            'dist_coeffs': [0, 0, 0, 0, 0],
            'homography_matrix': np.eye(3).tolist(),
            'image_size': [1280, 720],
            'pixel_to_meter_ratio': 0.1  # 10 cm per pixel
        }
        return CameraCalibration(cal_data)
        
    def test_calculator_initialization(self):
        """Test velocity calculator initialization"""
        calibration = self.create_test_calibration()
        
        calculator = VelocityCalculator(
            calibration=calibration,
            fps=30.0,
            smoothing_window=5,
            velocity_smoothing='moving_average'
        )
        
        assert calculator.fps == 30.0
        assert calculator.smoothing_window == 5
        assert calculator.velocity_smoothing == 'moving_average'
        assert len(calculator.track_histories) == 0
        
    def test_ground_contact_point(self):
        """Test ground contact point calculation"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration)
        
        bbox = [100, 100, 200, 200]
        contact_point = calculator._get_ground_contact_point(bbox)
        
        # Should be bottom center
        expected = (150.0, 200.0)
        assert contact_point == expected
        
    def test_track_update_new_track(self):
        """Test updating with new track"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration, min_track_length=2)
        
        # First update - should create track but not calculate velocity
        bbox = [100, 100, 200, 200]
        result = calculator.update_track(track_id=1, bbox=bbox, frame_number=0)
        
        assert result is None  # Not enough history yet
        assert 1 in calculator.track_histories
        assert len(calculator.track_histories[1].positions_world) == 1
        
    def test_track_update_velocity_calculation(self):
        """Test velocity calculation with sufficient history"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration, min_track_length=2, fps=30.0)
        
        # Add multiple points to simulate movement
        bboxes = [
            [100, 100, 200, 200],  # Frame 0
            [110, 100, 210, 200],  # Frame 1 - moved 10 pixels right
            [120, 100, 220, 200],  # Frame 2 - moved another 10 pixels
        ]
        
        results = []
        for i, bbox in enumerate(bboxes):
            result = calculator.update_track(track_id=1, bbox=bbox, frame_number=i)
            results.append(result)
            
        # Should have velocity measurement from second update onward
        assert results[0] is None  # First point
        assert results[1] is not None  # Should have velocity now
        assert isinstance(results[1], VelocityMeasurement)
        assert results[1].track_id == 1
        assert results[1].velocity_ms > 0  # Should detect movement
        
    def test_velocity_calculation_methods(self):
        """Test different velocity calculation methods"""
        calibration = self.create_test_calibration()
        
        methods = ['moving_average', 'kalman', 'polynomial']
        
        for method in methods:
            calculator = VelocityCalculator(
                calibration, 
                min_track_length=2,
                velocity_smoothing=method
            )
            
            # Add consistent movement
            for i in range(5):
                bbox = [100 + i*10, 100, 200 + i*10, 200]
                result = calculator.update_track(track_id=1, bbox=bbox, frame_number=i)
                
            # Should calculate reasonable velocity
            assert result is not None
            assert result.velocity_ms > 0
            assert 0 <= result.confidence <= 1
            
    def test_outlier_removal(self):
        """Test outlier detection and removal"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration, outlier_threshold=2.0)
        
        # Create positions with one outlier
        positions = np.array([
            [0, 0], [1, 1], [2, 2], [100, 100], [3, 3], [4, 4]  # [100, 100] is outlier
        ])
        timestamps = np.array([0, 1, 2, 3, 4, 5])
        
        clean_pos, clean_time = calculator._remove_outliers(positions, timestamps)
        
        # Should remove the outlier
        assert len(clean_pos) < len(positions)
        # The outlier [100, 100] should be removed
        assert not any(np.array_equal(pos, [100, 100]) for pos in clean_pos)
        
    def test_track_cleanup(self):
        """Test cleanup of old tracks"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration, max_track_age=1.0)
        
        # Add some tracks
        calculator.update_track(1, [100, 100, 200, 200], 0, timestamp=0.0)
        calculator.update_track(2, [300, 300, 400, 400], 0, timestamp=0.0)
        
        assert len(calculator.track_histories) == 2
        
        # Cleanup after 2 seconds (older than max_track_age)
        calculator.cleanup_old_tracks(current_timestamp=2.0)
        
        assert len(calculator.track_histories) == 0
        
    def test_performance_stats(self):
        """Test performance statistics"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration)
        
        # No stats initially
        stats = calculator.get_performance_stats()
        assert stats == {}
        
        # Add some calculation times
        calculator.calculation_times = [0.01, 0.02, 0.015]
        
        stats = calculator.get_performance_stats()
        assert 'mean_calculation_time' in stats
        assert 'total_calculations' in stats
        assert stats['total_calculations'] == 3
        
    def test_reset_functions(self):
        """Test reset functionality"""
        calibration = self.create_test_calibration()
        calculator = VelocityCalculator(calibration)
        
        # Add some data
        calculator.update_track(1, [100, 100, 200, 200], 0)
        calculator.calculation_times = [0.01, 0.02]
        
        # Reset stats
        calculator.reset_stats()
        assert calculator.calculation_times == []
        
        # Reset tracks
        calculator.reset_all_tracks()
        assert len(calculator.track_histories) == 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason=f"Dependencies not available: {IMPORT_ERROR if not IMPORTS_AVAILABLE else ''}")
class TestHelperFunctions:
    """Test helper functions"""
    
    def test_create_simple_calibration(self):
        """Test creating simple calibration"""
        calibration = create_simple_calibration(
            image_size=(1920, 1080),
            pixel_to_meter_ratio=0.05
        )
        
        assert isinstance(calibration, CameraCalibration)
        assert calibration.image_size == (1920, 1080)
        assert calibration.pixel_to_meter_ratio == 0.05
        
    def test_create_velocity_calculator(self):
        """Test creating velocity calculator from file"""
        # Create temporary calibration file
        cal_data = {
            'camera_matrix': np.eye(3).tolist(),
            'dist_coeffs': [0, 0, 0, 0, 0],
            'homography_matrix': np.eye(3).tolist(),
            'image_size': [1280, 720],
            'pixel_to_meter_ratio': 0.1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(cal_data, f)
            temp_path = f.name
            
        try:
            calculator = create_velocity_calculator(temp_path, fps=25.0)
            assert isinstance(calculator, VelocityCalculator)
            assert calculator.fps == 25.0
        finally:
            Path(temp_path).unlink()


def test_module_structure():
    """Test module structure without heavy dependencies"""
    module_path = Path(__file__).parent.parent.parent / "src" / "velocity" / "velocity_calculator.py"
    assert module_path.exists()
    
    # Check that the file contains expected class definitions
    content = module_path.read_text()
    assert "class VelocityCalculator" in content
    assert "class CameraCalibration" in content
    assert "class VelocityMeasurement" in content
    assert "def create_simple_calibration" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])