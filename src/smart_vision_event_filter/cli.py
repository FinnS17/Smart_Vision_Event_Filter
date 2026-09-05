import argparse

import cv2


from smart_vision_event_filter.detection import detect_motion_boxes
from smart_vision_event_filter.events import MotionEventTracker
from smart_vision_event_filter.reporting import save_events_to_json
from smart_vision_event_filter.clips import export_event_clips

FRAME_DELAY = 5  # Wartezeit für die Tastatureingabe in Millisekunden.

def main():

    parser = argparse.ArgumentParser(
        description="Detects motion events in a video file."
    )
    parser.add_argument(
        "video_path",
        help="path to video file"
    )
    parser.add_argument(
        "--max-gap-frames",
        type=int,
        default=10,
        help="maximum number of consecutive frames without motion inside an event"
    )
    parser.add_argument(
        "--output",
        default="outputs/motion_events.json",
        help="path for the generated JSON event report"
    )
    parser.add_argument(
        "--clips-dir",
        help="directory for exported event clips"
    )
    args = parser.parse_args()


    video_path = args.video_path
    max_gap_frames = args.max_gap_frames
    json_output_path = args.output

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file at: {video_path}")
        raise SystemExit
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Zu Beginn gibt es noch kein vorheriges Bild, mit dem wir vergleichen könnten.
    previous_gray = None

    tracker = MotionEventTracker(max_gap_frames=max_gap_frames)
    frame_number = -1
    completed_events = []

    # Das Video wird Frame für Frame verarbeitet, bis es endet oder `q` gedrückt wird.
    while True:
        # `ret` ist True, wenn ein Frame erfolgreich gelesen wurde.
        ret, frame = cap.read()
        if not ret: # end of video or error
            break

        frame_number += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            # Der erste Frame dient nur als Referenz; erst ab Frame zwei kann eine
            # Veränderung zwischen zwei Zeitpunkten berechnet werden.
            previous_gray = gray
            continue

        motion_boxes = detect_motion_boxes(previous_gray, gray)
        has_motion = bool(motion_boxes) # at least one movement
        completed_event = tracker.update(frame_number=frame_number, has_motion=has_motion)
        if completed_event is not None:
            completed_events.append(completed_event)
            print(f"Motion event: frames "
                  f"{completed_event.start_frame}-{completed_event.end_frame}")

        if motion_boxes:
            # Aus allen Einzelrechtecken wird ein gemeinsames Gesamt-Rechteck gebildet.
            min_x = min(x for x, y, w, h in motion_boxes)
            min_y = min(y for x, y, w, h in motion_boxes)
            max_x = max(x + w for x, y, w, h in motion_boxes)
            max_y = max(y + h for x, y, w, h in motion_boxes)

            # Das gemeinsame Rechteck und der Text machen den Frame für Menschen lesbar.
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "Motion Detected",
                (min_x, min_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Aktuellen, bereits markierten Frame im Fenster anzeigen.
        cv2.imshow("Motion Detection", frame)

        if cv2.waitKey(FRAME_DELAY) & 0xFF == ord("q"):
            break

        # Erst am Ende wird der aktuelle Frame zur Referenz für die nächste Runde.
        previous_gray = gray

    final_event = tracker.finish() # close last occuring event
    if final_event is not None:
        completed_events.append(final_event)
        print(
            f"Motion event: frames "
            f"{final_event.start_frame}-{final_event.end_frame}"
        )

    cap.release()
    cv2.destroyAllWindows()

    save_events_to_json(completed_events, str(json_output_path), fps)
    print(f"Saved {len(completed_events)} events to {json_output_path}")
    if args.clips_dir is not None:
        clip_paths = export_event_clips(video_path, completed_events, args.clips_dir)
        print(f"Exportetd {len(clip_paths)} clips to {args.clips_dir}")

if __name__ == "__main__":
    main()