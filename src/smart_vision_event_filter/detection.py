import cv2

THRESHOLD_VALUE = 60  # Small brightness changes do not count as motion.
DILATE_ITERATIONS = 5  # Join nearby changed pixels into larger areas.
MIN_AREA = 8000  # Ignore small areas of image noise.


def detect_motion_boxes(previous_gray, current_gray):
    """Find large motion areas by comparing two grayscale video frames."""

    # diff contains the brightness change at each pixel.
    diff = cv2.absdiff(previous_gray, current_gray)

    # Turn changed pixels into white areas, then join nearby areas.
    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        # OpenCV gives each box as (x, y, width, height).
        x, y, width, height = cv2.boundingRect(contour)
        motion_boxes.append((x, y, width, height))

    return motion_boxes
