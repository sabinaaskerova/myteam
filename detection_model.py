import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class DetectionModel:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the detection model with the specified model path and confidence threshold.

        Args:
            model_path (str): Path to the TFLite model.
            confidence_threshold (float): Minimum confidence score to consider a detection valid.
        """
        self.detection_result_list = []
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            score_threshold=confidence_threshold,
            result_callback=self._visualize_callback,
            max_results=20
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def _visualize_callback(self, result: vision.ObjectDetectorResult, 
                            output_image: mp.Image, timestamp_ms: int):
        """
        Callback function to store detection results, needed by ObjectDetectorOptions.

        Args:
            result (vision.ObjectDetectorResult): The detection result.
            output_image (mp.Image): The output image with detections.
            timestamp_ms (int): The timestamp of the detection.
        """
        result.timestamp_ms = timestamp_ms
        self.detection_result_list.append(result)

    def detect(self, mp_image, counter):
        """
        Run object detection.

        Args:
            mp_image (mp.Image): The input image for detection.
            counter (int): The frame counter.
        """
        self.detector.detect_async(mp_image, counter)

    def get_results(self):
        results = self.detection_result_list.copy()
        self.detection_result_list.clear()
        return results

    def close(self):
        self.detector.close()