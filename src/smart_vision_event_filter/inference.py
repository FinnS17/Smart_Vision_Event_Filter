from dataclasses import dataclass

from ultralytics import YOLO

from smart_vision_event_filter import config

@dataclass
class ObjectDetection:
    """Keep one YOLO result in a simple form for the rest of the project."""

    class_name: str
    confidence: float
    bounding_box: tuple[int, int, int, int]


def load_object_detector(model_name: str = config.YOLO_MODEL_NAME) -> YOLO:
    """Load the pretrained YOLO model once before processing frames."""
    return YOLO(model_name)


def detect_objects(model: YOLO, frame, confidence_threshold: float = config.YOLO_CONFIDENCE_THRESHOLD) -> list[ObjectDetection]:
    """Detect objects in one frame and return simple Python objects."""
    results = model.predict(source=frame, imgsz=config.IMAGE_SIZE, conf=confidence_threshold, verbose=False, device=config.DEVICE)
    detections = []
    # We passed one frame, so results[0] belongs to that frame.
    for box in results[0].boxes:
        # Example: class ID 0 may map to the name 'person'.
        class_id = int(box.cls.item())
        class_name = model.names[class_id]
        confidence = float(box.conf.item())
        # YOLO returns corners in original-frame coordinates: (x1, y1, x2, y2).
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        detection = ObjectDetection(class_name=class_name, confidence=confidence, bounding_box=(x1,y1,x2,y2))
        detections.append(detection)

    return detections


def overlaps_motion(detection: ObjectDetection, motion_boxes: list[tuple[int, int, int, int]]) -> bool:
    """Check whether a YOLO box overlaps any motion box."""
    x1, y1, x2, y2 = detection.bounding_box

    for motion_x, motion_y, width, height in motion_boxes:
        # Motion uses (x, y, width, height), so find its opposite corner.
        motion_x2 = motion_x + width
        motion_y2 = motion_y + height
        # The boxes overlap only when both axes overlap.
        overlaps_x = x1 < motion_x2 and x2 > motion_x
        overlaps_y = y1 < motion_y2 and y2 > motion_y

        if overlaps_x and overlaps_y:
            return True

    return False
