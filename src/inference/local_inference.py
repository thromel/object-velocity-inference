"""
Complete local inference pipeline for vehicle velocity estimation.
Designed for flexibility across different deployment scenarios.
"""

import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import argparse
import sys
from collections import deque

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core.video_input import VideoSource, VideoSourceFactory
    from models.yolo_detector import YOLOv9Detector, Detection
    from tracking.deepsort_tracker import DeepSORTTracker, Track
    from velocity.velocity_calculator import VelocityCalculator, VelocityMeasurement, CameraCalibration
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    COMPONENT_ERROR = str(e)


@dataclass
class FrameResult:
    """
    Results from processing a single frame
    """
    frame_number: int
    timestamp: float
    detections: List[Detection]
    tracks: List[Track]
    velocities: Dict[int, VelocityMeasurement]
    processing_time: float
    fps: float


@dataclass
class InferenceStats:
    """
    Performance statistics for inference
    """
    total_frames: int
    total_time: float
    avg_fps: float
    detection_times: List[float]
    tracking_times: List[float]
    velocity_times: List[float]
    avg_detection_time: float
    avg_tracking_time: float
    avg_velocity_time: float


class FPSCounter:
    """
    Utility class for FPS calculation
    """
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)
        
    def update(self):
        """Add current timestamp"""
        self.timestamps.append(time.time())
        
    def get_fps(self) -> float:
        """Calculate current FPS"""
        if len(self.timestamps) < 2:
            return 0.0
            
        time_diff = self.timestamps[-1] - self.timestamps[0]
        if time_diff <= 0:
            return 0.0
            
        return (len(self.timestamps) - 1) / time_diff


class LocalInferenceEngine:
    """
    Complete local inference pipeline for vehicle velocity estimation.
    Designed for flexibility across different deployment scenarios.
    """
    
    def __init__(self, config: Union[str, Path, Dict[str, Any]]):
        """
        Initialize inference engine from configuration.
        
        Args:
            config: Configuration file path or dictionary
        """
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = self._load_config(config)
        else:
            self.config = config
            
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.detector: Optional[YOLOv9Detector] = None
        self.tracker: Optional[DeepSORTTracker] = None
        self.velocity_calculator: Optional[VelocityCalculator] = None
        self.video_source: Optional[VideoSource] = None
        
        # Performance monitoring
        self.fps_counter = FPSCounter()
        self.inference_stats = InferenceStats(
            total_frames=0,
            total_time=0.0,
            avg_fps=0.0,
            detection_times=[],
            tracking_times=[],
            velocity_times=[],
            avg_detection_time=0.0,
            avg_tracking_time=0.0,
            avg_velocity_time=0.0
        )
        
        # Initialize components if available
        if COMPONENTS_AVAILABLE:
            self._init_components()
        else:
            self.logger.error(f"Components not available: {COMPONENT_ERROR}")
            
    def _load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file"""
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
            
        with open(config_file, 'r') as f:
            if config_file.suffix.lower() == '.json':
                config = json.load(f)
            elif config_file.suffix.lower() in ['.yml', '.yaml']:
                import yaml
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")
                
        # Validate required sections
        required_sections = ['detector', 'tracker', 'velocity']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        return config
        
    def _setup_logging(self):
        """Setup logging configuration"""
        log_config = self.config.get('logging', {})
        
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(level=level, format=format_str)
        
    def _init_components(self):
        """Initialize all inference components"""
        try:
            # Initialize detector
            detector_config = self.config['detector']
            self.detector = YOLOv9Detector(
                model_path=detector_config['model_path'],
                device=detector_config.get('device', 'auto'),
                conf_threshold=detector_config.get('conf_threshold', 0.5),
                iou_threshold=detector_config.get('iou_threshold', 0.45),
                target_classes=detector_config.get('target_classes'),
                img_size=detector_config.get('img_size', 640),
                half_precision=detector_config.get('half_precision', True)
            )
            
            # Load detector model
            if not self.detector.load_model():
                raise RuntimeError("Failed to load detection model")
                
            # Initialize tracker
            tracker_config = self.config['tracker']
            self.tracker = DeepSORTTracker(
                feature_extractor_path=tracker_config.get('feature_extractor_path'),
                max_cosine_distance=tracker_config.get('max_cosine_distance', 0.2),
                nn_budget=tracker_config.get('nn_budget', 100),
                max_age=tracker_config.get('max_age', 30),
                n_init=tracker_config.get('n_init', 3),
                max_iou_distance=tracker_config.get('max_iou_distance', 0.7),
                device=tracker_config.get('device', 'auto')
            )
            
            # Initialize velocity calculator
            velocity_config = self.config['velocity']
            self.velocity_calculator = VelocityCalculator(
                calibration=velocity_config['calibration_file'],
                fps=velocity_config.get('fps', 30.0),
                smoothing_window=velocity_config.get('smoothing_window', 5),
                min_track_length=velocity_config.get('min_track_length', 3),
                max_track_age=velocity_config.get('max_track_age', 2.0),
                velocity_smoothing=velocity_config.get('velocity_smoothing', 'kalman')
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
            
    def run_inference(self, 
                     source: Union[str, Dict[str, Any]],
                     output_path: Optional[str] = None,
                     display: bool = True,
                     save_stats: bool = True,
                     max_frames: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Main inference loop supporting various input sources.
        
        Args:
            source: Video source (file path, camera index, or config dict)
            output_path: Path to save output video
            display: Whether to display results
            save_stats: Whether to save performance statistics
            max_frames: Maximum number of frames to process
            
        Returns:
            Dictionary with results and statistics
        """
        if not COMPONENTS_AVAILABLE:
            self.logger.error("Cannot run inference: components not available")
            return None
            
        try:
            # Setup video source
            if isinstance(source, str):
                if source.isdigit():
                    # Camera index
                    source_config = {
                        'type': 'camera',
                        'index': int(source),
                        'source_id': f'camera_{source}'
                    }
                elif Path(source).exists():
                    # File path
                    source_config = {
                        'type': 'file', 
                        'path': source,
                        'source_id': f'file_{Path(source).stem}'
                    }
                else:
                    # Assume RTSP URL
                    source_config = {
                        'type': 'rtsp',
                        'url': source,
                        'source_id': 'rtsp_stream'
                    }
            else:
                source_config = source
                
            self.video_source = VideoSourceFactory.create_source(source_config)
            
            # Start video source
            if not self.video_source.start():
                raise RuntimeError("Failed to start video source")
                
            # Get video properties
            properties = self.video_source.get_properties()
            fps = properties['fps']
            
            # Setup video writer if saving output
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(
                    output_path, fourcc, fps, 
                    (properties['width'], properties['height'])
                )
                
            # Setup display window
            if display:
                cv2.namedWindow('Vehicle Velocity Tracking', cv2.WINDOW_AUTOSIZE)
                
            # Main processing loop
            frame_count = 0
            results = []
            start_time = time.time()
            
            self.logger.info("Starting inference loop")
            
            while True:
                # Get next frame
                ret, frame = self.video_source.get_frame(timeout=1.0)
                
                if not ret or frame is None:
                    break
                    
                # Process frame
                frame_start = time.time()
                result = self.process_frame(frame, frame_count)
                processing_time = time.time() - frame_start
                
                # Update FPS counter
                self.fps_counter.update()
                current_fps = self.fps_counter.get_fps()
                
                # Add timing info to result
                result.processing_time = processing_time
                result.fps = current_fps
                
                results.append(result)
                
                # Visualize results
                vis_frame = None
                if display or output_path:
                    vis_frame = self.visualize_results(frame, result)
                    self._add_performance_overlay(vis_frame, processing_time, current_fps)
                    
                # Display frame
                if display and vis_frame is not None:
                    cv2.imshow('Vehicle Velocity Tracking', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Reset tracking
                        self.tracker.reset_tracker()
                        self.velocity_calculator.reset_all_tracks()
                        
                # Save frame
                if writer and vis_frame is not None:
                    writer.write(vis_frame)
                    
                frame_count += 1
                
                # Check max frames limit
                if max_frames and frame_count >= max_frames:
                    break
                    
                # Periodic cleanup
                if frame_count % 100 == 0:
                    current_time = frame_count / fps
                    self.velocity_calculator.cleanup_old_tracks(current_time)
                    
            # Cleanup
            total_time = time.time() - start_time
            self._finalize_stats(total_time, frame_count)
            
            self.video_source.stop()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
                
            self.logger.info(f"Processed {frame_count} frames in {total_time:.2f}s "
                           f"(avg {frame_count/total_time:.1f} FPS)")
            
            # Save statistics
            final_results = {
                'frame_results': [asdict(r) for r in results],
                'stats': asdict(self.inference_stats),
                'config': self.config
            }
            
            if save_stats:
                self._save_results(final_results, output_path)
                
            return final_results
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
        finally:
            # Ensure cleanup
            if hasattr(self, 'video_source') and self.video_source:
                self.video_source.stop()
            if 'writer' in locals() and writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
                
    def process_frame(self, frame: np.ndarray, frame_number: int) -> FrameResult:
        """
        Process single frame through complete pipeline.
        
        Args:
            frame: Input frame
            frame_number: Frame number
            
        Returns:
            FrameResult with detections, tracks, and velocities
        """
        timestamp = frame_number / self.config['velocity'].get('fps', 30.0)
        
        # Initialize result
        result = FrameResult(
            frame_number=frame_number,
            timestamp=timestamp,
            detections=[],
            tracks=[],
            velocities={},
            processing_time=0.0,
            fps=0.0
        )
        
        try:
            # Step 1: Detection
            t1 = time.time()
            detections = self.detector.detect(frame)
            detection_time = time.time() - t1
            self.inference_stats.detection_times.append(detection_time)
            result.detections = detections
            
            # Convert detections to format expected by tracker
            detection_dicts = []
            for det in detections:
                detection_dicts.append({
                    'bbox': det.bbox,
                    'confidence': det.confidence,
                    'class_id': det.class_id,
                    'class_name': det.class_name
                })
            
            # Step 2: Tracking
            t2 = time.time()
            tracks = self.tracker.update(detection_dicts, frame)
            tracking_time = time.time() - t2
            self.inference_stats.tracking_times.append(tracking_time)
            result.tracks = tracks
            
            # Step 3: Velocity calculation
            t3 = time.time()
            velocities = {}
            for track in tracks:
                velocity = self.velocity_calculator.update_track(
                    track.track_id,
                    track.bbox,
                    frame_number,
                    timestamp
                )
                if velocity is not None:
                    velocities[track.track_id] = velocity
                    
            result.velocities = velocities
            velocity_time = time.time() - t3
            self.inference_stats.velocity_times.append(velocity_time)
            
        except Exception as e:
            self.logger.error(f"Frame processing failed: {e}")
            
        return result
        
    def visualize_results(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """
        Create visualization with bounding boxes, IDs, and velocities.
        
        Args:
            frame: Original frame
            result: Processing results
            
        Returns:
            Visualized frame
        """
        vis_frame = frame.copy()
        
        # Draw tracks
        for track in result.tracks:
            bbox = [int(x) for x in track.bbox]
            track_id = track.track_id
            
            # Get velocity if available
            velocity = result.velocities.get(track_id)
            
            # Choose color based on velocity
            if velocity is None:
                color = (128, 128, 128)  # Gray for no velocity
                velocity_text = "N/A"
            else:
                speed_kmh = velocity.velocity_kmh
                if speed_kmh < 30:
                    color = (0, 255, 0)      # Green for slow
                elif speed_kmh < 60:
                    color = (0, 165, 255)    # Orange for medium
                else:
                    color = (0, 0, 255)      # Red for fast
                velocity_text = f"{speed_kmh:.1f} km/h"
                
            # Draw bounding box
            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Create label
            label = f"ID: {track_id} | {velocity_text}"
            if velocity and hasattr(velocity, 'confidence'):
                label += f" ({velocity.confidence:.2f})"
                
            # Draw label with background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_frame,
                         (bbox[0], bbox[1] - label_size[1] - 4),
                         (bbox[0] + label_size[0], bbox[1]),
                         color, -1)
            cv2.putText(vis_frame, label,
                       (bbox[0], bbox[1] - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
        return vis_frame
        
    def _add_performance_overlay(self, frame: np.ndarray, processing_time: float, fps: float):
        """Add performance information overlay to frame"""
        height, width = frame.shape[:2]
        
        # Performance text
        perf_text = [
            f"FPS: {fps:.1f}",
            f"Processing: {processing_time*1000:.1f}ms",
            f"Tracks: {len(self.tracker.tracks) if self.tracker and hasattr(self.tracker, 'tracks') else 0}",
            f"Frame: {self.inference_stats.total_frames}"
        ]
        
        # Draw background rectangle
        text_height = 20 * len(perf_text) + 10
        cv2.rectangle(frame, (10, 10), (250, text_height), (0, 0, 0), -1)
        
        # Draw text
        for i, text in enumerate(perf_text):
            y_pos = 30 + i * 20
            cv2.putText(frame, text, (15, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
    def _finalize_stats(self, total_time: float, total_frames: int):
        """Finalize performance statistics"""
        self.inference_stats.total_frames = total_frames
        self.inference_stats.total_time = total_time
        self.inference_stats.avg_fps = total_frames / total_time if total_time > 0 else 0
        
        if self.inference_stats.detection_times:
            self.inference_stats.avg_detection_time = np.mean(self.inference_stats.detection_times)
        if self.inference_stats.tracking_times:
            self.inference_stats.avg_tracking_time = np.mean(self.inference_stats.tracking_times)
        if self.inference_stats.velocity_times:
            self.inference_stats.avg_velocity_time = np.mean(self.inference_stats.velocity_times)
            
    def _save_results(self, results: Dict[str, Any], output_path: Optional[str]):
        """Save results to file"""
        if output_path:
            results_path = Path(output_path).with_suffix('.json')
        else:
            results_path = Path('inference_results.json')
            
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Results saved to {results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        detector_stats = self.detector.get_performance_stats() if self.detector else {}
        tracker_stats = self.tracker.get_performance_stats() if self.tracker else {}
        velocity_stats = self.velocity_calculator.get_performance_stats() if self.velocity_calculator else {}
        
        return {
            'inference': asdict(self.inference_stats),
            'detector': detector_stats,
            'tracker': tracker_stats,
            'velocity': velocity_stats
        }


def main():
    """Command line interface for local inference"""
    parser = argparse.ArgumentParser(description='Vehicle velocity inference')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--source', required=True, help='Video source (file, camera, or RTSP URL)')
    parser.add_argument('--output', help='Output video path')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process')
    parser.add_argument('--save-stats', action='store_true', default=True, help='Save statistics')
    
    args = parser.parse_args()
    
    try:
        # Create inference engine
        engine = LocalInferenceEngine(args.config)
        
        # Run inference
        results = engine.run_inference(
            source=args.source,
            output_path=args.output,
            display=not args.no_display,
            save_stats=args.save_stats,
            max_frames=args.max_frames
        )
        
        if results:
            print("\nInference completed successfully!")
            print(f"Processed {results['stats']['total_frames']} frames")
            print(f"Average FPS: {results['stats']['avg_fps']:.1f}")
            
        else:
            print("Inference failed!")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())