"""
Unit tests for video input handling components.
"""

import pytest
import numpy as np
import cv2
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.core.video_input import (
    VideoSource, RTSPSource, FileSource, CameraSource, VideoSourceFactory
)


class MockVideoSource(VideoSource):
    """Mock video source for testing base functionality"""
    
    def __init__(self, source_id: str, frame_generator=None):
        super().__init__(source_id)
        self.frame_generator = frame_generator or self._default_frame_generator
        self.connected = False
        
    def connect(self) -> bool:
        self.connected = True
        return True
        
    def read_frame(self):
        if not self.connected:
            return False, None
        return next(self.frame_generator)
        
    def _default_frame_generator(self):
        """Generate test frames"""
        frame_count = 0
        while True:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frame_count += 1
            if frame_count <= 10:
                yield True, frame
            else:
                yield False, None


class TestVideoSource:
    """Test base VideoSource functionality"""
    
    def test_video_source_initialization(self):
        """Test VideoSource initialization"""
        source = MockVideoSource("test_source")
        
        assert source.source_id == "test_source"
        assert source.frame_queue.maxsize == 30
        assert not source.is_running
        assert source.frame_count == 0
        
    def test_video_source_start_stop(self):
        """Test starting and stopping video source"""
        source = MockVideoSource("test_source")
        
        # Test start
        assert source.start()
        assert source.is_running
        time.sleep(0.1)  # Allow thread to start
        
        # Test getting frames
        ret, frame = source.get_frame(timeout=1.0)
        assert ret
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        
        # Test stop
        source.stop()
        assert not source.is_running
        
    def test_video_source_properties(self):
        """Test getting video source properties"""
        source = MockVideoSource("test_source")
        source.start()
        time.sleep(0.1)
        
        properties = source.get_properties()
        
        assert properties['source_id'] == "test_source"
        assert properties['fps'] == 30.0
        assert properties['width'] == 1280
        assert properties['height'] == 720
        assert properties['is_running']
        
        source.stop()
        
    def test_frame_buffer_overflow(self):
        """Test frame buffer behavior when full"""
        def fast_frame_generator():
            while True:
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                yield True, frame
                
        source = MockVideoSource("test_source", fast_frame_generator())
        source.frame_queue = source.frame_queue.__class__(maxsize=5)  # Small buffer
        
        source.start()
        time.sleep(0.5)  # Let buffer fill
        
        # Buffer should not exceed maxsize
        assert source.frame_queue.qsize() <= 5
        
        source.stop()


class TestRTSPSource:
    """Test RTSP source functionality"""
    
    @patch('cv2.VideoCapture')
    def test_rtsp_connection_success(self, mock_cv2):
        """Test successful RTSP connection"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        source = RTSPSource("rtsp_test", "rtsp://test.url")
        
        assert source.connect()
        assert source.fps == 30.0
        assert source.width == 1920
        assert source.height == 1080
        
    @patch('cv2.VideoCapture')
    def test_rtsp_connection_failure(self, mock_cv2):
        """Test RTSP connection failure and retry"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_cv2.return_value = mock_cap
        
        source = RTSPSource("rtsp_test", "rtsp://invalid.url", reconnect_attempts=2)
        
        assert not source.connect()
        assert mock_cv2.call_count == 2  # Should retry
        
    @patch('cv2.VideoCapture')
    def test_rtsp_frame_reading(self, mock_cv2):
        """Test reading frames from RTSP"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        mock_cv2.return_value = mock_cap
        
        source = RTSPSource("rtsp_test", "rtsp://test.url")
        source.connect()
        
        ret, frame = source.read_frame()
        assert ret
        assert frame is not None
        assert frame.shape == (480, 640, 3)


class TestFileSource:
    """Test file source functionality"""
    
    def create_test_video(self, duration_seconds: float = 1.0) -> str:
        """Create a test video file"""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))
        
        frames_to_write = int(30 * duration_seconds)
        for i in range(frames_to_write):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            writer.write(frame)
            
        writer.release()
        return temp_path
        
    def test_file_source_connection(self):
        """Test file source connection"""
        video_path = self.create_test_video()
        
        try:
            source = FileSource("file_test", video_path)
            assert source.connect()
            assert source.fps > 0
            assert source.width == 640
            assert source.height == 480
            assert source.total_frames > 0
            
        finally:
            Path(video_path).unlink()
            
    def test_file_source_nonexistent_file(self):
        """Test file source with nonexistent file"""
        source = FileSource("file_test", "nonexistent.mp4")
        assert not source.connect()
        
    def test_file_source_loop(self):
        """Test file source looping"""
        video_path = self.create_test_video(duration_seconds=0.1)  # Very short video
        
        try:
            source = FileSource("file_test", video_path, loop=True)
            source.connect()
            
            # Read more frames than the video contains
            frames_read = 0
            for _ in range(10):
                ret, frame = source.read_frame()
                if ret:
                    frames_read += 1
                    
            # Should have looped and read frames
            assert frames_read > source.total_frames
            
        finally:
            Path(video_path).unlink()


class TestCameraSource:
    """Test camera source functionality"""
    
    @patch('cv2.VideoCapture')
    def test_camera_connection(self, mock_cv2):
        """Test camera connection"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 1280,
            cv2.CAP_PROP_FRAME_HEIGHT: 720,
            cv2.CAP_PROP_FPS: 30.0
        }.get(prop, 0)
        mock_cv2.return_value = mock_cap
        
        source = CameraSource("camera_test", camera_index=0)
        
        assert source.connect()
        assert source.width == 1280
        assert source.height == 720
        assert source.fps == 30.0
        
    @patch('cv2.VideoCapture')
    def test_camera_custom_resolution(self, mock_cv2):
        """Test camera with custom resolution"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cv2.return_value = mock_cap
        
        source = CameraSource(
            "camera_test", 
            camera_index=0, 
            resolution=(1920, 1080),
            fps=60.0
        )
        
        assert source.connect()
        
        # Verify settings were applied
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        mock_cap.set.assert_any_call(cv2.CAP_PROP_FPS, 60.0)


class TestVideoSourceFactory:
    """Test video source factory"""
    
    def test_create_rtsp_source(self):
        """Test creating RTSP source"""
        config = {
            'type': 'rtsp',
            'url': 'rtsp://test.url',
            'source_id': 'rtsp_cam'
        }
        
        source = VideoSourceFactory.create_source(config)
        
        assert isinstance(source, RTSPSource)
        assert source.source_id == 'rtsp_cam'
        assert source.rtsp_url == 'rtsp://test.url'
        
    def test_create_file_source(self):
        """Test creating file source"""
        config = {
            'type': 'file',
            'path': 'test.mp4',
            'source_id': 'file_source',
            'loop': True
        }
        
        source = VideoSourceFactory.create_source(config)
        
        assert isinstance(source, FileSource)
        assert source.source_id == 'file_source'
        assert str(source.file_path) == 'test.mp4'
        assert source.loop
        
    def test_create_camera_source(self):
        """Test creating camera source"""
        config = {
            'type': 'camera',
            'index': 1,
            'source_id': 'usb_cam',
            'resolution': [1920, 1080],
            'fps': 60.0
        }
        
        source = VideoSourceFactory.create_source(config)
        
        assert isinstance(source, CameraSource)
        assert source.source_id == 'usb_cam'
        assert source.camera_index == 1
        assert source.resolution == (1920, 1080)
        assert source.target_fps == 60.0
        
    def test_invalid_source_type(self):
        """Test creating source with invalid type"""
        config = {
            'type': 'invalid',
            'source_id': 'test'
        }
        
        with pytest.raises(ValueError, match="Unsupported source type"):
            VideoSourceFactory.create_source(config)
            
    def test_auto_generated_source_id(self):
        """Test auto-generated source ID"""
        config = {
            'type': 'camera',
            'index': 0
        }
        
        source = VideoSourceFactory.create_source(config)
        
        assert source.source_id.startswith('source_')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])