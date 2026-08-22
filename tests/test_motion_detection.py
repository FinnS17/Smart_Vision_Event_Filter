import numpy as np

from smart_vision_event_filter.detection import detect_motion_boxes


def test_returns_no_boxes_for_identical_frames():
    frame = np.zeros((100,100), dtype=np.uint8)
    boxes = detect_motion_boxes(frame, frame)

    assert boxes == []

def test_returns_box_for_large_changed_region():
    previous_frame = np.zeros((200,200), dtype=np.uint8)
    current_frame = previous_frame.copy()

    current_frame[40:160, 50:170] = 255
    boxes = detect_motion_boxes(previous_frame, current_frame)

    assert len(boxes) == 1

    x, y, width, height = boxes[0]

    assert x <= 50
    assert y <= 40
    assert x + width >= 170
    assert y + height >= 160