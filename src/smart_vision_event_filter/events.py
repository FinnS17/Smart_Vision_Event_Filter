from dataclasses import dataclass


@dataclass
class MotionEvent:
    """saves an event"""
    start_frame: int
    end_frame: int | None = None

    @property
    def is_active(self) -> bool:
        """Return whether the motion event is still open."""
        return self.end_frame is None

    def close(self, end_frame: int) -> None:
        """Close the event by saving its final motion frame."""
        if end_frame < self.start_frame:
            raise ValueError("end_frame cannot be before start_frame")
        self.end_frame = end_frame


class MotionEventTracker:
    """Memory for active events"""

    def __init__(self, max_gap_frames: int = 0) -> None:
        """Create a tracker that can tolerate short gaps without motion."""
        self.current_event: MotionEvent | None = None # aktuelle Event oder None
        self.max_gap_frames = max_gap_frames # erlaubte Lückengröße
        self.gap_frames = 0 # aktuelle Lückengröße
        self.last_motion_frame: int | None = None # letzer Frame mit Bewegung

    def update(self, frame_number: int, has_motion: bool) -> MotionEvent | None:
        """Update the tracker with the motion result of one video frame."""
        # called for every frame
        if has_motion:
            self.gap_frames = 0 # aktuelle Lückengröße bei erkannter Bewegung zurücksetzen
            self.last_motion_frame = frame_number # bei jedem Frame mit Motion sich ihn merken
            if self.current_event is None: # start new event if no current event
                self.current_event = MotionEvent(start_frame=frame_number)
            return None

        if self.current_event is None: # if no current event, nothing to close
            return None

        self.gap_frames += 1
        if self.gap_frames <= self.max_gap_frames:
            return None

        # wenn keine Motion, es aktuell laufendes Event gibt und Gap frames überschritten -> Event beenden
        self.current_event.close(end_frame=self.last_motion_frame)

        completed_event = self.current_event
        self.current_event = None
        self.gap_frames = 0
        self.last_motion_frame = None
        return completed_event


    def finish(self) -> MotionEvent | None:
        """Close and return the active event when video processing finishes."""
        # eg. current_event = MotionEvent(start_frame=100, end_frame=None)
        # last_motion_frame = 102
        # gap_frames = 0

        if self.current_event is None:
            return None

        self.current_event.close(end_frame=self.last_motion_frame)

        completed_event = self.current_event
        self.current_event = None
        self.gap_frames = 0
        self.last_motion_frame = None

        return completed_event

