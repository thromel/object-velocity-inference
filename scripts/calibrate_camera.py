#!/usr/bin/env python3
"""
Camera calibration utility for vehicle velocity measurement.
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def collect_calibration_points(image_path: str) -> dict:
    """
    Interactive tool to collect calibration points from an image.
    
    Args:
        image_path: Path to calibration image
        
    Returns:
        Dictionary with image and world coordinate pairs
    """
    if not CV2_AVAILABLE:
        print("OpenCV not available. Please install opencv-python.")
        return {}
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return {}
        
    print("Calibration Point Collection")
    print("=" * 40)
    print("Instructions:")
    print("1. Click on known points in the image")
    print("2. Enter the real-world coordinates for each point")
    print("3. Press 'q' when done (minimum 4 points needed)")
    print("4. Press 'r' to reset points")
    print()
    
    image_points = []
    world_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            image_points.append([x, y])
            
            # Draw point
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{len(image_points)}", (x+10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow('Calibration', img)
            
            # Get world coordinates
            print(f"Point {len(image_points)}: Image coordinates ({x}, {y})")
            while True:
                try:
                    world_x = float(input("Enter real-world X coordinate (meters): "))
                    world_y = float(input("Enter real-world Y coordinate (meters): "))
                    world_points.append([world_x, world_y])
                    print(f"Added point: Image({x}, {y}) -> World({world_x}, {world_y})")
                    print()
                    break
                except ValueError:
                    print("Please enter valid numbers")
    
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', mouse_callback)
    cv2.imshow('Calibration', img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset
            image_points.clear()
            world_points.clear()
            img = cv2.imread(image_path)
            cv2.imshow('Calibration', img)
            print("Points reset")
            
    cv2.destroyAllWindows()
    
    if len(image_points) < 4:
        print("Error: Need at least 4 points for calibration")
        return {}
        
    return {
        'image_points': image_points,
        'world_points': world_points
    }


def calculate_homography(image_points: list, world_points: list) -> np.ndarray:
    """
    Calculate homography matrix from corresponding points.
    
    Args:
        image_points: List of image coordinates
        world_points: List of world coordinates
        
    Returns:
        3x3 homography matrix
    """
    img_pts = np.array(image_points, dtype=np.float32)
    world_pts = np.array(world_points, dtype=np.float32)
    
    # Calculate homography
    homography, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC)
    
    return homography


def create_calibration_file(output_path: str,
                          homography: np.ndarray,
                          image_size: tuple,
                          camera_info: dict = None) -> None:
    """
    Create calibration JSON file.
    
    Args:
        output_path: Output file path
        homography: Homography matrix
        image_size: Image dimensions (width, height)
        camera_info: Additional camera information
    """
    # Default camera matrix (approximation)
    fx = fy = max(image_size) * 0.7  # Rough estimate
    cx, cy = image_size[0] / 2, image_size[1] / 2
    
    calibration_data = {
        "camera_matrix": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],  # Assume no distortion
        "homography_matrix": homography.tolist(),
        "image_size": list(image_size),
        "ground_plane_normal": [0.0, 0.0, 1.0],
        "ground_plane_point": [0.0, 0.0, 0.0],
        "pixel_to_meter_ratio": None,  # Will use homography instead
        "calibration_info": camera_info or {
            "description": "Camera calibration from manual point correspondence",
            "calibration_method": "homography",
            "calibration_date": "auto-generated",
            "notes": "Generated using calibrate_camera.py script"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
        
    print(f"Calibration saved to: {output_path}")


def create_simple_calibration(output_path: str,
                            pixel_to_meter_ratio: float,
                            image_size: tuple,
                            camera_info: dict = None) -> None:
    """
    Create simple calibration using pixel-to-meter ratio.
    
    Args:
        output_path: Output file path
        pixel_to_meter_ratio: Meters per pixel
        image_size: Image dimensions (width, height)
        camera_info: Additional camera information
    """
    calibration_data = {
        "camera_matrix": [
            [1.0, 0.0, image_size[0]/2],
            [0.0, 1.0, image_size[1]/2],
            [0.0, 0.0, 1.0]
        ],
        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        "homography_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ],
        "image_size": list(image_size),
        "ground_plane_normal": [0.0, 0.0, 1.0],
        "ground_plane_point": [0.0, 0.0, 0.0],
        "pixel_to_meter_ratio": pixel_to_meter_ratio,
        "calibration_info": camera_info or {
            "description": "Simple calibration using pixel-to-meter ratio",
            "calibration_method": "simple_scaling",
            "pixel_to_meter_ratio": pixel_to_meter_ratio,
            "calibration_date": "auto-generated",
            "notes": "Generated using calibrate_camera.py script"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(calibration_data, f, indent=2)
        
    print(f"Simple calibration saved to: {output_path}")


def main():
    """Main calibration function"""
    parser = argparse.ArgumentParser(
        description="Camera calibration for vehicle velocity measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive calibration from image
  python scripts/calibrate_camera.py --image frame.jpg --output my_calibration.json
  
  # Simple calibration with known pixel-to-meter ratio
  python scripts/calibrate_camera.py --simple --ratio 0.05 --size 1920 1080 --output simple_cal.json
        """
    )
    
    parser.add_argument(
        '--image',
        help='Calibration image path (for interactive calibration)'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output calibration file path'
    )
    
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Create simple calibration using pixel-to-meter ratio'
    )
    
    parser.add_argument(
        '--ratio',
        type=float,
        help='Pixel-to-meter ratio (for simple calibration)'
    )
    
    parser.add_argument(
        '--size',
        nargs=2,
        type=int,
        metavar=('WIDTH', 'HEIGHT'),
        help='Image size in pixels (for simple calibration)'
    )
    
    args = parser.parse_args()
    
    if args.simple:
        # Simple calibration mode
        if not args.ratio or not args.size:
            print("Error: Simple calibration requires --ratio and --size arguments")
            return 1
            
        create_simple_calibration(
            args.output,
            args.ratio,
            tuple(args.size)
        )
        
    else:
        # Interactive calibration mode
        if not args.image:
            print("Error: Interactive calibration requires --image argument")
            return 1
            
        if not CV2_AVAILABLE:
            print("Error: OpenCV not available for interactive calibration")
            print("Install with: pip install opencv-python")
            return 1
            
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            return 1
            
        print("Starting interactive calibration...")
        points_data = collect_calibration_points(args.image)
        
        if not points_data:
            print("Calibration cancelled or failed")
            return 1
            
        # Calculate homography
        homography = calculate_homography(
            points_data['image_points'],
            points_data['world_points']
        )
        
        # Get image size
        img = cv2.imread(args.image)
        height, width = img.shape[:2]
        
        # Create calibration file
        create_calibration_file(
            args.output,
            homography,
            (width, height)
        )
        
        print(f"Calibration completed with {len(points_data['image_points'])} points")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())