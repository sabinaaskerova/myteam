import argparse
from detection_system import DetectionSystem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientdet_lite2.tflite',
                      help='Path to the detection model')
    parser.add_argument('--confidence', type=float, default=0.5,
                      help='Minimum confidence threshold')
    parser.add_argument('--camera', type=int, default=0,
                      help='Camera device ID')
    args = parser.parse_args()
    camera = DetectionSystem(args.model, args.confidence)
    camera.run_camera(args.camera)

if __name__ == '__main__':
    main()