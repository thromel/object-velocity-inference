# Vehicle Velocity Inference System

A production-ready system for inferring vehicle velocities from video streams using YOLOv9 detection and DeepSORT tracking. This system combines computer vision and deep learning to provide accurate real-time velocity measurements for traffic monitoring, safety analysis, and transportation research.

## Features

- **Real-time Processing**: 15+ FPS on modern hardware with GPU acceleration
- **Accurate Detection**: YOLOv9-based vehicle detection with production optimizations
- **Robust Tracking**: DeepSORT multi-object tracking with appearance features
- **Precise Velocity**: Camera calibration and temporal smoothing for accurate measurements
- **Flexible Input**: Support for video files, live cameras, and RTSP streams
- **Production Ready**: Comprehensive error handling, monitoring, and deployment tools
- **Configurable**: Scenario-specific configurations for different deployment environments

## System Architecture

The system consists of four main components:

1. **Video Input Handler**: Manages various video sources with threading and buffering
2. **YOLOv9 Detector**: Vehicle detection with GPU optimization and batch processing
3. **DeepSORT Tracker**: Multi-object tracking with consistent ID assignment
4. **Velocity Calculator**: Real-world velocity calculation using camera calibration

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/object-velocity-inference.git
cd object-velocity-inference
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download model weights (example):
```bash
# Create models directory
mkdir -p models/pretrained

# Download YOLOv9 weights (replace with actual download)
# wget -O models/pretrained/yolov9_vehicles.pt https://example.com/yolov9_vehicles.pt
```

### Basic Usage

Run inference on a video file:
```bash
python scripts/run_inference.py --source path/to/video.mp4
```

Run on camera feed:
```bash
python scripts/run_inference.py --source 0
```

Run on RTSP stream:
```bash
python scripts/run_inference.py --source rtsp://camera.ip/stream
```

## Configuration

The system uses JSON configuration files for different scenarios:

### Available Configurations

- `configs/default_config.json`: General-purpose configuration
- `configs/scenarios/highway_config.json`: Optimized for highway monitoring
- `configs/scenarios/urban_config.json`: Optimized for urban intersections
- `configs/scenarios/edge_config.json`: Lightweight config for edge devices

### Using Different Configurations

```bash
python scripts/run_inference.py --source video.mp4 --config configs/scenarios/highway_config.json
```

## Camera Calibration

Accurate velocity measurement requires camera calibration. The system provides tools for both interactive and simple calibration:

### Interactive Calibration

Use a reference image with known ground truth points:
```bash
python scripts/calibrate_camera.py --image calibration_frame.jpg --output my_calibration.json
```

### Simple Calibration

For quick setup with approximate measurements:
```bash
python scripts/calibrate_camera.py --simple --ratio 0.05 --size 1920 1080 --output simple_calibration.json
```

## Project Structure

```
object-velocity-inference/
├── src/                          # Source code
│   ├── core/                     # Core components
│   │   └── video_input.py        # Video source handling
│   ├── models/                   # Detection models
│   │   └── yolo_detector.py      # YOLOv9 detector
│   ├── tracking/                 # Tracking components
│   │   └── deepsort_tracker.py   # DeepSORT tracker
│   ├── velocity/                 # Velocity calculation
│   │   └── velocity_calculator.py # Velocity calculator
│   └── inference/                # Inference pipeline
│       └── local_inference.py    # Main inference engine
├── configs/                      # Configuration files
│   ├── default_config.json       # Default configuration
│   ├── scenarios/                # Scenario-specific configs
│   └── calibration/              # Camera calibrations
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── scripts/                      # Utility scripts
│   ├── run_inference.py          # Main inference script
│   └── calibrate_camera.py       # Calibration utility
├── models/                       # Model storage
│   ├── pretrained/               # Pre-trained models
│   └── fine_tuned/               # Fine-tuned models
└── docs/                         # Documentation
    └── architecture.md           # Detailed architecture
```

## Configuration Options

### Detector Configuration
```json
{
  "detector": {
    "model_path": "models/pretrained/yolov9_vehicles.pt",
    "device": "auto",
    "conf_threshold": 0.5,
    "iou_threshold": 0.45,
    "target_classes": [2, 3, 5, 7],
    "img_size": 640,
    "half_precision": true
  }
}
```

### Tracker Configuration
```json
{
  "tracker": {
    "max_cosine_distance": 0.2,
    "nn_budget": 100,
    "max_age": 30,
    "n_init": 3,
    "max_iou_distance": 0.7
  }
}
```

### Velocity Configuration
```json
{
  "velocity": {
    "calibration_file": "configs/calibration/default_calibration.json",
    "fps": 30.0,
    "smoothing_window": 5,
    "velocity_smoothing": "kalman"
  }
}
```

## Performance Optimization

### GPU Optimization
- Enable FP16 inference for 2x speedup
- Use batch processing for multiple streams
- Optimize CUDA memory usage

### CPU Optimization
- Use OpenCV optimizations
- Enable multi-threading
- Reduce input resolution for edge devices

### Memory Management
- Automatic track cleanup
- Frame buffer management
- Efficient feature caching

## Development

### Running Tests

Run all tests:
```bash
python -m pytest tests/ -v
```

Run specific test categories:
```bash
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
```

### Code Quality

The project includes tools for maintaining code quality:
- Type hints throughout
- Comprehensive error handling
- Performance monitoring
- Logging and debugging support

## Deployment Scenarios

### Edge Deployment
- Lightweight model variants
- CPU-optimized configurations
- Reduced memory footprint
- Local processing only

### Cloud Deployment
- High-performance GPU processing
- Multiple stream handling
- Scalable architecture
- Remote monitoring

### Hybrid Deployment
- Edge preprocessing
- Cloud aggregation
- Bandwidth optimization
- Fault tolerance

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Verify model file exists
   - Check CUDA compatibility
   - Ensure sufficient memory

2. **Poor Detection Performance**
   - Adjust confidence thresholds
   - Verify lighting conditions
   - Check camera positioning

3. **Inaccurate Velocities**
   - Recalibrate camera
   - Verify FPS settings
   - Check tracking consistency

### Performance Issues

1. **Low FPS**
   - Enable GPU acceleration
   - Reduce input resolution
   - Use model optimization

2. **High Memory Usage**
   - Reduce batch size
   - Lower tracking history
   - Enable memory cleanup

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting guide
- Review the architecture documentation
- Open an issue on GitHub

## Citation

If you use this system in your research, please cite:

```bibtex
@software{vehicle_velocity_inference,
  title={Vehicle Velocity Inference System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/object-velocity-inference}
}
```