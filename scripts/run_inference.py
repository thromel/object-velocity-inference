#!/usr/bin/env python3
"""
Example script for running vehicle velocity inference.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from inference.local_inference import LocalInferenceEngine
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Error importing components: {e}")
    print("Please install required dependencies from requirements.txt")


def main():
    """Main function for running inference"""
    if not IMPORTS_AVAILABLE:
        print("Components not available. Please check installation.")
        return 1
        
    parser = argparse.ArgumentParser(
        description="Run vehicle velocity inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on video file with default config
  python scripts/run_inference.py --source video.mp4
  
  # Run on camera with urban config
  python scripts/run_inference.py --source 0 --config configs/scenarios/urban_config.json
  
  # Run on RTSP stream with output video
  python scripts/run_inference.py --source rtsp://camera.ip/stream --output results.mp4
  
  # Run for specific number of frames without display
  python scripts/run_inference.py --source video.mp4 --max-frames 100 --no-display
        """
    )
    
    parser.add_argument(
        '--source', 
        required=True,
        help='Video source: file path, camera index (0, 1, ...), or RTSP URL'
    )
    
    parser.add_argument(
        '--config',
        default='configs/default_config.json',
        help='Configuration file path (default: configs/default_config.json)'
    )
    
    parser.add_argument(
        '--output',
        help='Output video file path (optional)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable real-time display'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        help='Maximum number of frames to process'
    )
    
    parser.add_argument(
        '--save-stats',
        action='store_true',
        default=True,
        help='Save performance statistics (default: True)'
    )
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Available configs:")
        config_dir = Path("configs")
        if config_dir.exists():
            for config_file in config_dir.rglob("*.json"):
                print(f"  {config_file}")
        return 1
        
    try:
        print(f"Starting inference with config: {config_path}")
        print(f"Source: {args.source}")
        
        # Create inference engine
        engine = LocalInferenceEngine(str(config_path))
        
        # Run inference
        results = engine.run_inference(
            source=args.source,
            output_path=args.output,
            display=not args.no_display,
            save_stats=args.save_stats,
            max_frames=args.max_frames
        )
        
        if results:
            print("\n" + "="*50)
            print("INFERENCE COMPLETED SUCCESSFULLY")
            print("="*50)
            
            stats = results['stats']
            print(f"Total frames processed: {stats['total_frames']}")
            print(f"Total processing time: {stats['total_time']:.2f} seconds")
            print(f"Average FPS: {stats['avg_fps']:.1f}")
            print(f"Average detection time: {stats['avg_detection_time']*1000:.1f} ms")
            print(f"Average tracking time: {stats['avg_tracking_time']*1000:.1f} ms")
            print(f"Average velocity time: {stats['avg_velocity_time']*1000:.1f} ms")
            
            # Count vehicles with velocities
            vehicles_with_velocity = 0
            total_velocity_measurements = 0
            for frame_result in results['frame_results']:
                if frame_result['velocities']:
                    vehicles_with_velocity += len(frame_result['velocities'])
                    total_velocity_measurements += len(frame_result['velocities'])
                    
            print(f"Total velocity measurements: {total_velocity_measurements}")
            
            if args.output:
                print(f"Output video saved: {args.output}")
                
            print("\nPerformance summary saved to inference_results.json")
            
        else:
            print("Inference failed!")
            return 1
            
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 0
    except Exception as e:
        print(f"Error during inference: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())