import cv2

THRESHOLD_VALUE = 60  # Mindest-Helligkeitsänderung, die als Bewegung zählt.
DILATE_ITERATIONS = 5  # Verbindet nahe beieinanderliegende Bewegungs-Pixel.
MIN_AREA = 8000  # Kleinere Flächen gelten als Rauschen und werden ignoriert.


def detect_motion_boxes(previous_gray, current_gray):
    """Find large motion areas by comparing two grayscale video frames."""

    diff = cv2.absdiff(previous_gray, current_gray)

    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_AREA:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        motion_boxes.append((x, y, width, height))

    return motion_boxes
