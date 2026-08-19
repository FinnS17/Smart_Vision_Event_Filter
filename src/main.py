import cv2


THRESHOLD_VALUE = 60
DILATE_ITERATIONS = 5
MIN_AREA = 8000
FRAME_DELAY = 1

VIDEO_PATH = "videos/sample2.mov"


# --- load video -------
video_path = VIDEO_PATH
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f" Could not open video file at: {video_path}")
    raise SystemExit


previous_gray = None

while True:
    ret, frame = cap.read() # ret -> frame succesfully read?
    if not ret:
        break # leave loop if no frames left

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # -> (h, b)
    if previous_gray is None:
        previous_gray = gray
        continue # zur nächsten schleifenrunde springen
        
    diff = cv2.absdiff(previous_gray, gray)# calculate difference between current and previos frame
    _, thresh = cv2.threshold(diff, THRESHOLD_VALUE , 255, cv2.THRESH_BINARY) # diff > 25 -> 255; <25 -> 0
    dilated = cv2.dilate(thresh, None, iterations=DILATE_ITERATIONS)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # nur die äußeren konturen nhemne

    motion_boxes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue
        
        x, y, w, h = cv2.boundingRect(contour) # aus konturen recheck machen
        motion_boxes.append((x, y, w, h))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1) # rechteck einzeichnen

    if motion_boxes:
        min_x = min(x for x, y, w, h in motion_boxes)
        min_y = min(y for x, y, w, h in motion_boxes)
        max_x = max(x+w for x, y, w, h in motion_boxes)
        max_y = max(y+h for x, y, w, h in motion_boxes)

        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
        cv2.putText(frame, "Motion Detected", (min_x, min_y -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 2)

    cv2.imshow("Motion Detection", frame) # show current frame

    if cv2.waitKey(FRAME_DELAY) & 0xFF == ord("q"):
        break

    previous_gray = gray

cap.release() # cap -> zugriff auf video
cv2.destroyAllWindows()