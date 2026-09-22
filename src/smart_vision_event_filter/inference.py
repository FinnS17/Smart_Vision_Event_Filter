from dataclasses import dataclass

from ultralytics import YOLO

from smart_vision_event_filter import config

@dataclass
class ObjectDetection:
    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


def load_object_detector(model_name: str = config.YOLO_MODEL_NAME) -> YOLO:
    """Load pretrained YOLO object detection model"""
    return YOLO(model_name)


def detect_objects(model: YOLO, frame, confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD) -> list[ObjectDetection]:
    """Detect objects in one frame and return simplified detection results"""
    results = model.predict(source=frame, conf=confidence_threshold, verbose=False)
    detections = []
    for box in results[0].boxes: # we passed one frame, so the first result belongs to that frame
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detection = ObjectDetection(class_name=class_name, confidence=confidence, bounding_box=(x1,y1,x2,y2))
        detections.append(detection)
    return detections