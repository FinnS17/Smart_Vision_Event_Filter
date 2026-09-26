import argparse

import cv2

from smart_vision_event_filter.detection import detect_motion_boxes
from smart_vision_event_filter.events import MotionEventTracker
from smart_vision_event_filter.reporting import save_events_to_json
from smart_vision_event_filter.clips import export_event_clips


def main():
    """Run the complete motion detection pipeline from the command line."""

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
    parser.add_argument(
        "--clip-padding-seconds",
        type=float,
        default=1.0,
        help="seconds of context before and after each event clip"
    )
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        help="run object detection on frames containing motion",
    )
    parser.add_argument(
        "--target-class",
        help="export only events containting this object class"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="process the video without opening a window"
    )
    args = parser.parse_args()
    if args.target_class is not None and not args.enable_ai:
        parser.error("--target-class requires --enable-ai")

    # Without --enable-ai, the program only uses motion detection.
    object_detector = None
    if args.enable_ai:
        # Load YOLO only when the user enables AI.
        from smart_vision_event_filter.inference import load_object_detector, detect_objects, overlaps_motion
        object_detector = load_object_detector()


    video_path = args.video_path
    max_gap_frames = args.max_gap_frames
    json_output_path = args.output

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video file at: {video_path}")
        raise SystemExit
    fps = cap.get(cv2.CAP_PROP_FPS)

    # The first frame has no previous frame to compare with.
    previous_gray = None

    # The tracker remembers motion across frames and closes finished events.
    tracker = MotionEventTracker(max_gap_frames=max_gap_frames)
    frame_number = -1
    # Example: [MotionEvent(start_frame=10, end_frame=25), ...]
    completed_events = []

    # Process frames until the video ends or the user presses q.
    while True:
        # ret tells us whether OpenCV read a frame.
        ret, frame = cap.read()
        if not ret: # end of video or error
            break

        frame_number += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is None:
            # Keep the first frame as the reference for the next one.
            previous_gray = gray
            continue

        # Compare this frame with the previous one to find changed areas.
        motion_boxes = detect_motion_boxes(previous_gray, gray)
        # Example: [(x, y, width, height), ...]; [] means no motion.
        has_motion = bool(motion_boxes)

        # Start with no matching YOLO objects for this frame.
        relevant_detections = []
        if object_detector is not None and has_motion:
            # YOLO sees the whole frame: person, couch, etc.
            all_detections = detect_objects(object_detector, frame)
            for detection in all_detections:
                # Keep only objects whose box overlaps a motion area.
                if overlaps_motion(detection, motion_boxes):
                    relevant_detections.append(detection)

        # update returns a finished event when the quiet gap gets too long.
        completed_event = tracker.update(frame_number=frame_number, has_motion=has_motion)
        # update may have just created the event for this frame.
        if tracker.current_event is not None:
            for detection in relevant_detections:
                # Example: ['person', 'dog'], with each class stored once.
                if detection.class_name not in tracker.current_event.detected_objects:
                    tracker.current_event.detected_objects.append(detection.class_name)

        if completed_event is not None:
            # The same event keeps its start, end, and detected objects.
            completed_events.append(completed_event)
            print(f"Motion event: frames "
                  f"{completed_event.start_frame}-{completed_event.end_frame}")

        if motion_boxes and object_detector is None:
            # Show one combined motion box when AI is off.
            min_x = min(x for x, y, w, h in motion_boxes)
            min_y = min(y for x, y, w, h in motion_boxes)
            max_x = max(x + w for x, y, w, h in motion_boxes)
            max_y = max(y + h for x, y, w, h in motion_boxes)

            # Draw the motion result for the preview.
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 3)
            cv2.putText(frame, "Motion Detected", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)

        # In AI mode, show only objects that overlap motion.
        for detection in relevant_detections:
            x1, y1, x2, y2 = detection.bounding_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = (f"{detection.class_name} "
                        f"{detection.confidence:.2f}")
            cv2.putText(frame, label, (x1, max(y1 -10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if not args.no_preview:
            cv2.imshow("Motion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        previous_gray = gray

    # Close an event that is still active when the video stops.
    final_event = tracker.finish()
    if final_event is not None:
        completed_events.append(final_event)
        print(
            f"Motion event: frames "
            f"{final_event.start_frame}-{final_event.end_frame}"
        )

    cap.release()
    if not args.no_preview:
        cv2.destroyAllWindows()

    events_to_export = completed_events
    if args.target_class is not None:
        events_to_export = [event for event in completed_events if args.target_class in event.detected_objects]

    # Turn the finished event objects into a JSON report.
    save_events_to_json(events_to_export, str(json_output_path), fps)
    print(f"Saved {len(events_to_export)} events to {json_output_path}")
    if args.clips_dir is not None:
        clip_paths = export_event_clips(video_path, events_to_export, args.clips_dir, padding_seconds=args.clip_padding_seconds)
        print(f"Exportetd {len(clip_paths)} clips to {args.clips_dir}")

if __name__ == "__main__":
    main()
