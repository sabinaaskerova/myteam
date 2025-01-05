import cv2

# for visibility of labels inside the bounding box
MARGIN = 10  # Margin in pixels for the label inside the bounding box
ROW_SIZE = 10  # Row size in pixels for the label text
FONT_SIZE = 0.5
FONT_THICKNESS = 2
TEXT_COLOR = (0, 255, 0)  

# for FPS display
FPS_COLOR = (0, 0, 255)

class Visualizer:
    @staticmethod
    def visualize_detections(image, detection_result):
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
            category = detection.categories[0]
            confidence = round(category.score, 2)
            label = f'{category.category_name}: {confidence}'
            label_position = (bbox.origin_x + MARGIN, bbox.origin_y + MARGIN + ROW_SIZE)
            cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
        return image

    @staticmethod
    def add_fps(frame, fps):
        fps_text = f'FPS: {fps:.1f}'
        cv2.putText(frame, fps_text, (24, 20), cv2.FONT_HERSHEY_PLAIN, 1, FPS_COLOR, 1)
        return frame