# 🚀 Solovision

<div align="center">

<img src="assets/logo/logo.png" alt="Solovision Logo" width="200"/>

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

</div>

Solovision is a state-of-the-art real-time object tracking system that seamlessly integrates with ReID (Re-Identification) architecture. Built on top of YOLO object detection, it provides robust multi-object tracking capabilities with advanced features for identity preservation across frames.

<div align="center">
  <img src="assets/results/solovision_results.gif" alt="SoloVision Results">
</div>

## ✨ Key Features

- 🎯 **High-Performance Tracking**: Implements ByteTrack algorithm for reliable multi-object tracking
- 🔄 **ReID Integration**: Advanced re-identification capabilities for maintaining object identity
- 🚀 **Real-time Processing**: Optimized for real-time applications with efficient processing
- 📊 **Multiple Detection Backends**: Support for YOLOv8, YOLOv9, and other YOLO variants
- 💪 **Robust Motion Prediction**: Kalman filtering for smooth trajectory estimation
- 🎨 **Flexible Visualization**: Customizable visualization options for tracking results
- 🔧 **Easy-to-use CLI**: Simple command-line interface for quick deployment

## 🛠️ Installation

Install the solovision package in a Python>=3.9 environment.
```bash
pip install solovision
```

Install from source:

```bash
git clone https://github.com/AIEngineersDev/solovision.git
cd solovision
pip install .
```

Install in Dev
```bash
pip install poetry
poetry install
poetry shell
```

## 🚀 Quick Start

### Basic Usage

```python
from solovision import ByteTracker
from ultralytics import YOLO
import cv2

# Initialize tracker
tracker = ByteTracker(
    reid_weights="path/to/reid/weights",
    device="cuda",
    half=True
)

# Process video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Get detections from yolo
    model = YOLO('yolov8m.pt')
    detections = model.detect(frame)
    
    # Update tracker
    tracks = tracker.update(detections, frame)
    
    # Process tracking results
    for track in tracks:
        bbox = track[:4]
        track_id = track[4]
        # Draw or process tracking results
```

### Command Line Interface

```bash
# Track objects in a video
solovision track --source video_path --yolo-model yolov8n.pt --reid-model osnet_x1_0_msmt17.pt

# Track with custom settings
solovision track --source video_path --conf 0.25 --iou 0.45 --show --save --half \
                --show-trajectories --save-txt --save-crops --per-class \
                --classes 0 2 --device 0 --imgsz 640

# View all available CLI options
solovision track --help
```

## 🎯 ReID Models Support

Solovision supports various state-of-the-art ReID architectures:

- OSNet (x0.25, x0.5, x0.75, x1.0)
- OSNet-AIN
- OSNet-IBN
- ResNet (50, 101)
- CLIP-ReID

Check out the [Model Zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html) for pre-trained weights and performance benchmarks.

## 🔧 Advanced Features

- **Per-Class Tracking**: Enable separate tracking for different object classes
- **Feature History**: Maintain temporal appearance features for robust tracking
- **Camera Motion Compensation**: Automatic adjustment for camera movement
- **Multi-Camera Support**: Track objects across multiple camera views
- **Export Formats**: Save results in various formats (TXT, JSON, Video)

## 📊 Performance

- Runs at 30+ FPS on modern GPUs with YOLOv8n
- Support for half-precision (FP16) inference
- Optimized for both accuracy and speed
- Scalable for multi-camera deployments

## 🚀 Future Work

- Interactive Web Application (Coming Soon!)
  - Real-time tracking visualization
  - Model performance analytics
  - Easy configuration interface
  - Results visualization and export
  - Multiple video stream support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

```bibtex
@software{solovision2024,
  author = {Diddi, Dhruv and Mohammed, Zeeshaan},
  title = {Solovision: State-of-the-art Real-Time Object Tracking System},
  year = {2024},
  publisher = {GitHub},
  organization = {AIEngineersDev},
  url = {https://github.com/AIEngineersDev/solovision}
}
```

## 🙏 Acknowledgments

- ByteTrack algorithm implementation
- Ultralytics YOLO
- OSNet for ReID features
- BOXMOT
- FastReID

---
<p align="center">Made with ❤️ by Solo</p>
