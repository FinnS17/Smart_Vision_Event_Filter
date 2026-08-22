import numpy as np

from main import detect_motion_boxes


def test_returns_no_boxes_for_identical_frames():
    frame = np.zeros((100,100), dtype=np.uint8)
    boxes = detect_motion_boxes(frame, frame)

    assert boxes == []