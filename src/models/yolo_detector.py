"""
YOLOv9 detection module optimized for production vehicle detection.
Handles model loading, inference, and post-processing.
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
import time
from dataclasses import dataclass


@dataclass
class Detection:
    """
    Data class for detection results
    """
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class YOLOv9Detector:
    """
    YOLOv9 detection module optimized for production vehicle detection.
    Handles model loading, inference, and post-processing.
    """
    
    # COCO class names for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle', 
        5: 'bus',
        7: 'truck'
    }
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 target_classes: Optional[List[int]] = None,
                 img_size: int = 640,
                 half_precision: bool = True):
        """
        Initialize YOLOv9 detector with production optimizations.
        
        Args:
            model_path: Path to YOLOv9 model file
            device: Device to run inference on ('auto', 'cpu', 'cuda:0', etc.)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            target_classes: List of class IDs to detect (None = all vehicles)
            img_size: Input image size for model
            half_precision: Use FP16 for faster inference
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or list(self.VEHICLE_CLASSES.keys())
        self.img_size = img_size
        self.half_precision = half_precision
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.model_loaded = False
        
        # Performance tracking
        self.inference_times = []
        
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
            else:
                device = 'cpu'
                self.half_precision = False  # FP16 not supported on CPU
                
        device_obj = torch.device(device)
        self.logger.info(f"Using device: {device_obj}")
        
        return device_obj
        
    def load_model(self) -> bool:
        """
        Load YOLOv9 model with error handling and validation
        """
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
                
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Try to load with ultralytics first
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                self.model.to(self.device)
                
                # Configure model settings
                self.model.conf = self.conf_threshold
                self.model.iou = self.iou_threshold
                
                self.logger.info("Model loaded successfully with ultralytics")
                
            except ImportError:
                # Fallback to torch.load
                self.logger.warning("Ultralytics not available, using torch.load")
                self.model = torch.load(self.model_path, map_location=self.device)
                
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                    
            # Apply optimizations
            self._optimize_model()
            
            # Warm up model
            self._warmup_model()
            
            self.model_loaded = True
            self.logger.info("Model initialization complete")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
            
    def _optimize_model(self):
        """
        Apply production optimizations for faster inference
        """
        if self.model is None:
            return
            
        # Use half precision on GPU for 2x speedup
        if self.device.type == 'cuda' and self.half_precision:
            try:
                if hasattr(self.model, 'half'):
                    self.model.half()
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'half'):
                    self.model.model.half()
                self.logger.info("Enabled FP16 inference")
            except Exception as e:
                self.logger.warning(f"Failed to enable FP16: {e}")
                self.half_precision = False
                
        # Enable cudnn benchmarking for additional optimization
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            self.logger.info("Enabled cuDNN benchmarking")
            
    def _warmup_model(self, warmup_iterations: int = 5):
        """
        Warm up the model with dummy input for consistent performance
        """
        if self.model is None:
            return
            
        self.logger.info("Warming up model...")
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, (self.img_size, self.img_size, 3), dtype=np.uint8)
        
        # Run warmup iterations
        for i in range(warmup_iterations):
            try:
                _ = self.detect(dummy_input)
            except Exception as e:
                self.logger.warning(f"Warmup iteration {i+1} failed: {e}")
                
        self.logger.info("Model warmup complete")
        
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single frame.
        
        Args:
            frame: Input frame as numpy array (H, W, 3) in BGR format
            
        Returns:
            List of Detection objects
        """
        if not self.model_loaded:
            if not self.load_model():
                return []
                
        start_time = time.time()
        
        try:
            # Preprocess frame
            processed_frame = self._preprocess_frame(frame)
            
            # Run inference
            with torch.no_grad():
                results = self._run_inference(processed_frame)
                
            # Post-process results
            detections = self._postprocess_results(results, frame.shape)
            
            # Track inference time
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            # Keep only recent times for rolling average
            if len(self.inference_times) > 100:
                self.inference_times = self.inference_times[-100:]
                
            return detections
            
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []
            
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Batch processing for improved throughput.
        Essential for real-time multi-camera systems.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection lists, one per frame
        """
        if not self.model_loaded:
            if not self.load_model():
                return [[] for _ in frames]
                
        if not frames:
            return []
            
        start_time = time.time()
        
        try:
            # Preprocess all frames
            processed_frames = [self._preprocess_frame(frame) for frame in frames]
            
            # Stack into batch tensor
            if hasattr(self.model, 'predict'):
                # Ultralytics YOLO
                batch_results = []
                for processed_frame in processed_frames:
                    results = self.model.predict(processed_frame, verbose=False)
                    batch_results.append(results)
            else:
                # Custom batch processing for other model types
                batch_tensor = torch.stack(processed_frames)
                with torch.no_grad():
                    batch_results = self.model(batch_tensor)
                    
            # Post-process each frame's results
            all_detections = []
            for i, results in enumerate(batch_results):
                detections = self._postprocess_results(results, frames[i].shape)
                all_detections.append(detections)
                
            # Track batch inference time
            inference_time = time.time() - start_time
            avg_time_per_frame = inference_time / len(frames)
            self.inference_times.append(avg_time_per_frame)
            
            return all_detections
            
        except Exception as e:
            self.logger.error(f"Batch detection failed: {e}")
            return [[] for _ in frames]
            
    def _preprocess_frame(self, frame: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Prepare frame for YOLOv9 inference.
        Handles resizing, normalization, and tensor conversion.
        """
        # Resize while maintaining aspect ratio
        resized = self._letterbox_resize(frame, (self.img_size, self.img_size))
        
        # Convert BGR to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb = resized
            
        # Check if using ultralytics (expects numpy) or custom model (expects tensor)
        if hasattr(self.model, 'predict'):
            # Ultralytics expects numpy array
            return rgb
        else:
            # Custom model expects tensor
            normalized = rgb.astype(np.float32) / 255.0
            tensor = torch.from_numpy(normalized).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0).to(self.device)
            
            if self.half_precision and self.device.type == 'cuda':
                tensor = tensor.half()
                
            return tensor
            
    def _letterbox_resize(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image with padding to maintain aspect ratio
        """
        h, w = frame.shape[:2]
        target_h, target_w = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        
        # Place resized image in center
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        return padded
        
    def _run_inference(self, processed_input: Union[np.ndarray, torch.Tensor]):
        """Run model inference"""
        if hasattr(self.model, 'predict'):
            # Ultralytics YOLO
            results = self.model.predict(processed_input, verbose=False)
            return results[0] if results else None
        else:
            # Custom model
            return self.model(processed_input)
            
    def _postprocess_results(self, results, original_shape: Tuple[int, int, int]) -> List[Detection]:
        """
        Post-process model results into Detection objects
        """
        detections = []
        
        if results is None:
            return detections
            
        try:
            # Handle ultralytics results
            if hasattr(results, 'boxes') and results.boxes is not None:
                boxes = results.boxes
                
                for i in range(len(boxes)):
                    # Get box coordinates
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    cls_id = int(boxes.cls[i])
                    
                    # Filter by confidence and class
                    if conf < self.conf_threshold:
                        continue
                    if cls_id not in self.target_classes:
                        continue
                        
                    # Scale coordinates back to original image
                    scaled_box = self._scale_coordinates(box, original_shape)
                    
                    # Create detection
                    detection = Detection(
                        bbox=scaled_box.tolist(),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=self.VEHICLE_CLASSES.get(cls_id, f'class_{cls_id}')
                    )
                    
                    detections.append(detection)
                    
            else:
                # Handle custom model results (implement as needed)
                self.logger.warning("Custom model post-processing not implemented")
                
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            
        return detections
        
    def _scale_coordinates(self, box: np.ndarray, original_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Scale coordinates from model input size back to original image size
        """
        orig_h, orig_w = original_shape[:2]
        
        # Calculate scale factor (reverse of letterbox)
        scale = min(self.img_size / orig_w, self.img_size / orig_h)
        
        # Calculate padding
        pad_x = (self.img_size - orig_w * scale) / 2
        pad_y = (self.img_size - orig_h * scale) / 2
        
        # Remove padding and scale back
        box[0] = (box[0] - pad_x) / scale  # x1
        box[1] = (box[1] - pad_y) / scale  # y1
        box[2] = (box[2] - pad_x) / scale  # x2
        box[3] = (box[3] - pad_y) / scale  # y2
        
        # Clip to image bounds
        box[0] = max(0, min(box[0], orig_w))
        box[1] = max(0, min(box[1], orig_h))
        box[2] = max(0, min(box[2], orig_w))
        box[3] = max(0, min(box[3], orig_h))
        
        return box
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics"""
        if not self.inference_times:
            return {}
            
        times = np.array(self.inference_times)
        
        return {
            'mean_inference_time': float(np.mean(times)),
            'median_inference_time': float(np.median(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'std_inference_time': float(np.std(times)),
            'fps_estimate': float(1.0 / np.mean(times)) if np.mean(times) > 0 else 0.0,
            'total_inferences': len(self.inference_times)
        }
        
    def reset_stats(self):
        """Reset performance statistics"""
        self.inference_times = []
        
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            
        # Clear CUDA cache if using GPU
        if hasattr(self, 'device') and self.device.type == 'cuda':
            torch.cuda.empty_cache()


# Helper function for easy model creation
def create_detector(model_path: str, **kwargs) -> YOLOv9Detector:
    """
    Create and load a YOLOv9 detector
    
    Args:
        model_path: Path to model file
        **kwargs: Additional arguments for YOLOv9Detector
        
    Returns:
        Initialized YOLOv9Detector
    """
    detector = YOLOv9Detector(model_path, **kwargs)
    detector.load_model()
    return detector