"""
Basic tests to verify project structure without heavy dependencies.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def test_project_structure():
    """Test that required directories exist"""
    project_root = Path(__file__).parent.parent.parent
    
    required_dirs = [
        "src",
        "src/core",
        "src/models", 
        "src/tracking",
        "src/velocity",
        "tests",
        "configs",
        "models"
    ]
    
    for dir_path in required_dirs:
        assert (project_root / dir_path).exists(), f"Missing directory: {dir_path}"


def test_init_files():
    """Test that __init__.py files exist in Python packages"""
    project_root = Path(__file__).parent.parent.parent
    
    init_files = [
        "src/__init__.py",
        "src/core/__init__.py",
        "src/models/__init__.py",
        "src/tracking/__init__.py",
        "src/velocity/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        assert (project_root / init_file).exists(), f"Missing __init__.py: {init_file}"


def test_video_input_module_structure():
    """Test that video input module can be imported and has expected classes"""
    try:
        from core.video_input import VideoSource, VideoSourceFactory
        
        # Test that classes exist and are properly structured
        assert hasattr(VideoSource, 'connect')
        assert hasattr(VideoSource, 'read_frame')
        assert hasattr(VideoSource, 'start')
        assert hasattr(VideoSource, 'stop')
        
        assert hasattr(VideoSourceFactory, 'create_source')
        
    except ImportError as e:
        pytest.skip(f"Cannot import video_input module (missing dependencies): {e}")


def test_requirements_file():
    """Test that requirements.txt exists and has essential packages"""
    project_root = Path(__file__).parent.parent.parent
    requirements_file = project_root / "requirements.txt"
    
    assert requirements_file.exists(), "requirements.txt not found"
    
    content = requirements_file.read_text()
    
    # Check for essential packages
    essential_packages = [
        "torch",
        "opencv-python", 
        "numpy",
        "ultralytics",
        "pytest"
    ]
    
    for package in essential_packages:
        assert package in content, f"Missing package in requirements.txt: {package}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])