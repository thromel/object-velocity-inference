# Getting Started: Vehicle Velocity Inference

This guide will walk you through setting up the system and running velocity inference on your 1080p videos from scratch.

## Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional but recommended for faster processing)
- Sample 1080p video files of traffic

## Step 1: Install Dependencies

First, install all required Python packages:

```bash
# Navigate to project directory
cd object-velocity-inference

# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU-only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install ultralytics opencv-python numpy scipy
pip install filterpy scikit-image lap
pip install fastapi uvicorn redis websockets
pip install prometheus-client psutil
pip install pyyaml click python-dotenv tqdm
pip install pytest pytest-asyncio pytest-benchmark
pip install matplotlib seaborn jupyter
```

## Step 2: Download Pre-trained YOLOv9 Model

### Option A: Download YOLOv9 from Ultralytics (Recommended)

```bash
# Create models directory
mkdir -p models/pretrained

# Download YOLOv9c model (best balance of speed/accuracy)
python -c "
from ultralytics import YOLO
model = YOLO('yolov9c.pt')
model.export(format='pt')
import shutil
shutil.move('yolov9c.pt', 'models/pretrained/yolov9_vehicles.pt')
print('Model downloaded successfully!')
"
```

### Option B: Manual Download

If the above doesn't work, manually download:

```bash
# Create models directory
mkdir -p models/pretrained

# Download using wget or curl
wget -O models/pretrained/yolov9_vehicles.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt

# OR using curl
curl -L -o models/pretrained/yolov9_vehicles.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt
```

### Option C: Use YOLOv8 as Alternative

If YOLOv9 is not available:

```bash
python -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt for better accuracy
import shutil
shutil.move('yolov8n.pt', 'models/pretrained/yolov9_vehicles.pt')
print('YOLOv8 model downloaded as fallback!')
"
```

## Step 3: Verify Model Installation

Test that the model loads correctly:

```bash
python -c "
import sys
sys.path.append('src')
from models.yolo_detector import YOLOv9Detector

try:
    detector = YOLOv9Detector('models/pretrained/yolov9_vehicles.pt', device='cpu')
    success = detector.load_model()
    print(f'Model loading: {'SUCCESS' if success else 'FAILED'}')
except Exception as e:
    print(f'Error: {e}')
"
```

## Step 4: Prepare Your Video Files

Place your 1080p video files in a convenient location:

```bash
# Create a videos directory (optional)
mkdir -p data/sample_videos

# Copy your videos there
# cp /path/to/your/videos/*.mp4 data/sample_videos/
```

## Step 5: Quick Test Run

Let's run a quick test to make sure everything works:

```bash
# Test with a short video clip or first 100 frames
python scripts/run_inference.py \
    --source data/sample_videos/your_video.mp4 \
    --config configs/default_config.json \
    --max-frames 100 \
    --output test_output.mp4
```

## Step 6: Create Camera Calibration

For accurate velocity measurements, you need camera calibration. Here are your options:

### Option A: Quick Setup with Estimated Calibration

For initial testing, use approximate calibration:

```bash
# Create a simple calibration (adjust pixel_to_meter_ratio based on your camera setup)
python scripts/calibrate_camera.py \
    --simple \
    --ratio 0.05 \
    --size 1920 1080 \
    --output configs/calibration/my_calibration.json
```

**How to estimate pixel-to-meter ratio:**
- Look at your video and identify objects of known size (lane width ≈ 3.7m, car length ≈ 4.5m)
- Measure the object in pixels in your video
- Calculate: ratio = real_world_meters / pixels
- Example: If a 3.7m lane is 74 pixels wide, ratio = 3.7/74 = 0.05

### Option B: Interactive Calibration (More Accurate)

Extract a frame from your video for calibration:

```bash
# Extract a frame using ffmpeg (install if needed: brew install ffmpeg)
ffmpeg -i data/sample_videos/your_video.mp4 -vframes 1 -q:v 2 calibration_frame.jpg

# Run interactive calibration
python scripts/calibrate_camera.py \
    --image calibration_frame.jpg \
    --output configs/calibration/my_calibration.json
```

**In the interactive tool:**
1. Click on 4+ points with known real-world coordinates
2. Enter the actual coordinates in meters
3. Press 'q' when done

### Option C: Use Existing Calibration

Update an existing calibration file:

```bash
cp configs/calibration/default_calibration.json configs/calibration/my_calibration.json
# Edit the pixel_to_meter_ratio in my_calibration.json
```

## Step 7: Update Configuration

Create a custom configuration for your videos:

```bash
cp configs/default_config.json configs/my_config.json
```

Edit `configs/my_config.json` to point to your calibration:

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
  },
  "tracker": {
    "max_cosine_distance": 0.2,
    "nn_budget": 100,
    "max_age": 30,
    "n_init": 3,
    "max_iou_distance": 0.7,
    "device": "auto"
  },
  "velocity": {
    "calibration_file": "configs/calibration/my_calibration.json",
    "fps": 30.0,
    "smoothing_window": 5,
    "min_track_length": 3,
    "max_track_age": 2.0,
    "velocity_smoothing": "kalman"
  },
  "logging": {
    "level": "INFO"
  }
}
```

## Step 8: Run Full Inference

Now run inference on your full video:

```bash
python scripts/run_inference.py \
    --source data/sample_videos/your_video.mp4 \
    --config configs/my_config.json \
    --output results/output_with_velocities.mp4 \
    --save-stats
```

### For Multiple Videos

Process multiple videos in batch:

```bash
# Process all MP4 files in a directory
for video in data/sample_videos/*.mp4; do
    echo "Processing $video..."
    python scripts/run_inference.py \
        --source "$video" \
        --config configs/my_config.json \
        --output "results/$(basename "$video" .mp4)_results.mp4" \
        --save-stats
done
```

## Step 9: Understanding the Output

### Visual Output
- **Green boxes**: Slow vehicles (< 30 km/h)
- **Orange boxes**: Medium speed vehicles (30-60 km/h)  
- **Red boxes**: Fast vehicles (> 60 km/h)
- **Gray boxes**: Vehicles without velocity measurement

### Text Display
Each vehicle shows:
- **ID**: Unique tracking ID
- **Speed**: Velocity in km/h
- **Confidence**: Measurement confidence (0-1)

### Results File
The system saves detailed results in `inference_results.json`:

```json
{
  "frame_results": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "detections": [...],
      "tracks": [...],
      "velocities": {
        "1": {
          "track_id": 1,
          "velocity_kmh": 45.2,
          "velocity_ms": 12.6,
          "confidence": 0.89,
          "direction_angle": 15.3
        }
      }
    }
  ],
  "stats": {
    "total_frames": 1500,
    "avg_fps": 28.5,
    "avg_detection_time": 0.023
  }
}
```

## Step 10: Extract Velocity Data

To extract just the velocity data for analysis:

```bash
python -c "
import json
import csv

# Load results
with open('inference_results.json', 'r') as f:
    results = json.load(f)

# Extract velocities
velocities = []
for frame in results['frame_results']:
    for track_id, velocity_data in frame['velocities'].items():
        velocities.append({
            'frame': frame['frame_number'],
            'timestamp': frame['timestamp'],
            'track_id': track_id,
            'velocity_kmh': velocity_data['velocity_kmh'],
            'confidence': velocity_data['confidence']
        })

# Save to CSV
with open('velocities.csv', 'w', newline='') as f:
    if velocities:
        writer = csv.DictWriter(f, fieldnames=velocities[0].keys())
        writer.writeheader()
        writer.writerows(velocities)

print(f'Extracted {len(velocities)} velocity measurements to velocities.csv')
"
```

## Troubleshooting

### Common Issues

**1. Model Loading Error**
```bash
# Check if model file exists
ls -la models/pretrained/yolov9_vehicles.pt

# Try with CPU device
python scripts/run_inference.py --source video.mp4 --config configs/my_config.json
# In config, set "device": "cpu"
```

**2. Low Detection Performance**
```bash
# Lower confidence threshold in config
"conf_threshold": 0.3  # instead of 0.5

# Or try different model size
# Download yolov9m.pt or yolov9l.pt for better accuracy
```

**3. Inaccurate Velocities**
```bash
# Check your calibration - recalibrate with accurate measurements
python scripts/calibrate_camera.py --image frame.jpg --output new_calibration.json

# Verify FPS setting matches your video
ffprobe -v quiet -show_entries stream=r_frame_rate -of csv=p=0 your_video.mp4
```

**4. Out of Memory**
```bash
# Reduce image size in config
"img_size": 416  # instead of 640

# Or disable half precision
"half_precision": false
```

### Performance Optimization

**For Faster Processing:**
```json
{
  "detector": {
    "img_size": 416,
    "half_precision": true
  },
  "input": {
    "resize_width": 1280,
    "resize_height": 720,
    "frame_skip": 1
  }
}
```

**For Better Accuracy:**
```json
{
  "detector": {
    "conf_threshold": 0.3,
    "img_size": 832
  },
  "velocity": {
    "smoothing_window": 7,
    "min_track_length": 5
  }
}
```

## Next Steps

1. **Analyze Results**: Use the generated CSV files to analyze traffic patterns
2. **Tune Parameters**: Adjust confidence thresholds and tracking parameters for your specific videos
3. **Batch Processing**: Set up scripts to process multiple videos automatically
4. **Production Deployment**: Use the production deployment configs for real-time processing

## Example Complete Workflow

Here's a complete example from start to finish:

```bash
# 1. Setup
mkdir -p models/pretrained data/sample_videos results

# 2. Download model
python -c "from ultralytics import YOLO; YOLO('yolov9c.pt'); import shutil; shutil.move('yolov9c.pt', 'models/pretrained/yolov9_vehicles.pt')"

# 3. Copy your video
cp /path/to/your/traffic_video.mp4 data/sample_videos/

# 4. Create simple calibration
python scripts/calibrate_camera.py --simple --ratio 0.05 --size 1920 1080 --output configs/calibration/my_calibration.json

# 5. Create config
cp configs/default_config.json configs/my_config.json
# Edit to use your calibration file

# 6. Run inference
python scripts/run_inference.py \
    --source data/sample_videos/traffic_video.mp4 \
    --config configs/my_config.json \
    --output results/traffic_with_velocities.mp4

# 7. Check results
ls -la results/
cat inference_results.json | jq '.stats'
```

You should now have a video with velocity annotations and detailed measurement data!