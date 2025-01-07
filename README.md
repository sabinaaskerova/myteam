# Real-Time Object Recognition System

## Overview
This project is a real-time object recognition system developed using Mediapipe and OpenCV. It detects and classifies common objects (e.g., cups, phones) in a live video feed and displays the results with labels and confidence scores.

## Features
- **Real-Time Detection**: Process video streams in real-time.
- **Object Classification**: Classify detected objects using a pre-trained model.
- **FPS Display**: Show frames per second (FPS) for performance monitoring.



##  Download the Pre-trained Model
Download the EfficientDet-Lite2 TFLite model file:
```bash
wget https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float16/latest/efficientdet_lite2.tflite
```
Ensure the downloaded model file is located in the project root directory.


## Running the Project

### 1. Command-Line Arguments
The project supports customizable command-line arguments:
- `--model`: Path to the TFLite model file (default: `efficientdet_lite2.tflite`).
- `--confidence`: Confidence threshold for detections (default: `0.5`).
- `--camera`: ID of the camera device (default: `0`).

### 2. Run the Application
Execute the main script to start the object recognition system:
```bash
python main.py --model efficientdet_lite2.tflite --confidence 0.5 --camera 0
```



## Project Structure
```
project/
├── main.py                # Entry point of the application
├── detection_system.py    # Core system for processing detections
├── detection_model.py     # Model wrapper for Mediapipe object detection
├── visualization.py       # Utilities for drawing bounding boxes and FPS
├── efficientdet_lite2.tflite # Pre-trained model file
```
