import time
import cv2
import mediapipe as mp
from detection_model import DetectionModel
from visualization import Visualizer

class DetectionSystem:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the detection system with the specified model and confidence threshold.

        Args:
            model_path (str): Path to the TFLite model.
            confidence_threshold (float): Minimum confidence score to consider a detection valid.
        """
        self.detector = DetectionModel(model_path, confidence_threshold)
        self.counter = 0
        self.fps = 0
        self.start_time = time.time()
        self.fps_avg_frame_count = 10

    def process_frame(self, frame):
        """
        Process a single frame for object detection and return the annotated result.

        Args:
            frame: The input frame from the camera.

        Returns:
            The processed frame with visualized detections and FPS.
        """
        self.counter += 1 

        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection
        self.detector.detect(mp_image, self.counter)

        # Calculate FPS
        if self.counter % self.fps_avg_frame_count == 0:
            end_time = time.time()
            self.fps = self.fps_avg_frame_count / (end_time - self.start_time)
            self.start_time = time.time()
        # Visualize FPS in the frame
        frame = Visualizer.add_fps(frame, self.fps)

        # Get detection results and visualize them too
        results = self.detector.get_results()
        if results:
            frame = Visualizer.visualize_detections(frame, results[0])

        return frame

    def run_camera(self, camera_id: int = 0, width: int = 1280, height: int = 720):
        """
        Run the detection system on the camera feed.

        Args:
            camera_id (int): ID of the camera device.
            width (int): Width of the camera feed.
            height (int): Height of the camera feed.
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to read from camera")
                break

            # Mirror the image
            frame = cv2.flip(frame, 1)

            # Process the frame for object detection visualization
            processed_frame = self.process_frame(frame)

             # Display the processed frame
            cv2.imshow('Object Detection', processed_frame)

            # Exit on ESC key or window close
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Object Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()