"""
Video input handling for various sources including RTSP, files, and cameras.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from queue import Queue, Empty
from threading import Thread, Event
import logging
import time
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class VideoSource(ABC):
    """
    Abstract base class for all video sources.
    This design allows us to treat different video inputs uniformly,
    whether they're live cameras, files, or network streams.
    """
    
    def __init__(self, source_id: str, buffer_size: int = 30):
        self.source_id = source_id
        self.frame_queue = Queue(maxsize=buffer_size)
        self.is_running = False
        self.logger = logging.getLogger(f"VideoSource.{source_id}")
        self.capture_thread: Optional[Thread] = None
        self.stop_event = Event()
        self.frame_count = 0
        self.fps = 30.0
        self.width = 1280
        self.height = 720
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to video source"""
        pass
        
    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from source"""
        pass
        
    def start(self) -> bool:
        """Start the frame capture thread"""
        if not self.connect():
            return False
            
        self.is_running = True
        self.stop_event.clear()
        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        self.logger.info(f"Started video source: {self.source_id}")
        return True
        
    def stop(self):
        """Stop the frame capture thread"""
        self.is_running = False
        self.stop_event.set()
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
            
        self.logger.info(f"Stopped video source: {self.source_id}")
        
    def get_frame(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the next available frame from the buffer"""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except Empty:
            return False, None
            
    def get_properties(self) -> Dict[str, Any]:
        """Get video properties"""
        return {
            'source_id': self.source_id,
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'frame_count': self.frame_count,
            'is_running': self.is_running,
            'queue_size': self.frame_queue.qsize()
        }
        
    def _capture_loop(self):
        """
        Continuous capture loop running in separate thread.
        This prevents frame drops and ensures smooth processing.
        """
        consecutive_failures = 0
        max_failures = 10
        
        while self.is_running and not self.stop_event.is_set():
            try:
                ret, frame = self.read_frame()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    self.frame_count += 1
                    
                    # Drop old frames if buffer is full to maintain real-time processing
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except Empty:
                            pass
                            
                    self.frame_queue.put(frame)
                    
                else:
                    consecutive_failures += 1
                    self.logger.warning(
                        f"Failed to read frame from {self.source_id} "
                        f"(failure #{consecutive_failures})"
                    )
                    
                    if consecutive_failures >= max_failures:
                        self.logger.error(
                            f"Too many consecutive failures for {self.source_id}, stopping"
                        )
                        break
                        
                    time.sleep(0.1)  # Brief pause before retry
                    
            except Exception as e:
                self.logger.error(f"Error in capture loop for {self.source_id}: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    break
                    
                time.sleep(0.1)
                
        self.is_running = False
        self.logger.info(f"Capture loop ended for {self.source_id}")


class RTSPSource(VideoSource):
    """
    Handles RTSP streams with reconnection logic and error handling
    """
    
    def __init__(self, 
                 source_id: str, 
                 rtsp_url: str, 
                 reconnect_attempts: int = 3,
                 buffer_size: int = 30):
        super().__init__(source_id, buffer_size)
        self.rtsp_url = rtsp_url
        self.reconnect_attempts = reconnect_attempts
        self.cap: Optional[cv2.VideoCapture] = None
        
    def connect(self) -> bool:
        """
        Connect to RTSP stream with retry logic.
        Uses optimal OpenCV settings for network streams.
        """
        for attempt in range(self.reconnect_attempts):
            try:
                self.cap = cv2.VideoCapture(self.rtsp_url)
                
                # Optimize for real-time streaming
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                if self.cap.isOpened():
                    # Get stream properties
                    self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                    self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    self.logger.info(
                        f"Connected to RTSP stream: {self.source_id} "
                        f"({self.width}x{self.height} @ {self.fps} FPS)"
                    )
                    return True
                    
            except Exception as e:
                self.logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
                
        return False
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame with automatic reconnection on failure"""
        if self.cap is None or not self.cap.isOpened():
            if not self.connect():
                return False, None
                
        ret, frame = self.cap.read()
        
        if not ret:
            # Try to reconnect once
            self.logger.warning(f"Frame read failed for {self.source_id}, attempting reconnect")
            if self.connect():
                ret, frame = self.cap.read()
                
        return ret, frame
        
    def stop(self):
        """Stop RTSP stream and release resources"""
        super().stop()
        if self.cap:
            self.cap.release()
            self.cap = None


class FileSource(VideoSource):
    """
    Handles video file input with loop support
    """
    
    def __init__(self, 
                 source_id: str, 
                 file_path: str,
                 loop: bool = False,
                 buffer_size: int = 30):
        super().__init__(source_id, buffer_size)
        self.file_path = Path(file_path)
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.total_frames = 0
        
    def connect(self) -> bool:
        """Connect to video file"""
        if not self.file_path.exists():
            self.logger.error(f"Video file not found: {self.file_path}")
            return False
            
        try:
            self.cap = cv2.VideoCapture(str(self.file_path))
            
            if self.cap.isOpened():
                # Get video properties
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                self.logger.info(
                    f"Opened video file: {self.file_path.name} "
                    f"({self.width}x{self.height} @ {self.fps} FPS, {self.total_frames} frames)"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to open video file {self.file_path}: {e}")
            
        return False
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from file with loop support"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        
        if not ret and self.loop:
            # Reset to beginning of file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            self.logger.debug(f"Looped video file: {self.file_path.name}")
            
        return ret, frame
        
    def stop(self):
        """Stop file playback and release resources"""
        super().stop()
        if self.cap:
            self.cap.release()
            self.cap = None


class CameraSource(VideoSource):
    """
    Handles local camera input
    """
    
    def __init__(self, 
                 source_id: str, 
                 camera_index: int = 0,
                 resolution: Tuple[int, int] = (1280, 720),
                 fps: float = 30.0,
                 buffer_size: int = 30):
        super().__init__(source_id, buffer_size)
        self.camera_index = camera_index
        self.resolution = resolution
        self.target_fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        
    def connect(self) -> bool:
        """Connect to local camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if self.cap.isOpened():
                # Set camera properties
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Get actual properties
                self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or self.target_fps
                
                self.logger.info(
                    f"Connected to camera {self.camera_index}: "
                    f"({self.width}x{self.height} @ {self.fps} FPS)"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to connect to camera {self.camera_index}: {e}")
            
        return False
        
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from camera"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        return self.cap.read()
        
    def stop(self):
        """Stop camera capture and release resources"""
        super().stop()
        if self.cap:
            self.cap.release()
            self.cap = None


class VideoSourceFactory:
    """Factory for creating video sources"""
    
    @staticmethod
    def create_source(source_config: Dict[str, Any]) -> VideoSource:
        """
        Create a video source based on configuration
        
        Args:
            source_config: Dictionary containing source configuration
            
        Returns:
            VideoSource instance
            
        Example configurations:
            RTSP: {'type': 'rtsp', 'url': 'rtsp://...', 'source_id': 'cam1'}
            File: {'type': 'file', 'path': 'video.mp4', 'source_id': 'file1', 'loop': True}
            Camera: {'type': 'camera', 'index': 0, 'source_id': 'cam0'}
        """
        source_type = source_config.get('type', '').lower()
        source_id = source_config.get('source_id', f'source_{int(time.time())}')
        
        if source_type == 'rtsp':
            return RTSPSource(
                source_id=source_id,
                rtsp_url=source_config['url'],
                reconnect_attempts=source_config.get('reconnect_attempts', 3),
                buffer_size=source_config.get('buffer_size', 30)
            )
            
        elif source_type == 'file':
            return FileSource(
                source_id=source_id,
                file_path=source_config['path'],
                loop=source_config.get('loop', False),
                buffer_size=source_config.get('buffer_size', 30)
            )
            
        elif source_type == 'camera':
            return CameraSource(
                source_id=source_id,
                camera_index=source_config.get('index', 0),
                resolution=tuple(source_config.get('resolution', [1280, 720])),
                fps=source_config.get('fps', 30.0),
                buffer_size=source_config.get('buffer_size', 30)
            )
            
        else:
            raise ValueError(f"Unsupported source type: {source_type}")


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)