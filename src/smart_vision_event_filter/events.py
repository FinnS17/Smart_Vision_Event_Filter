from dataclasses import dataclass, field


@dataclass
class MotionEvent:
    """Store the time span and object classes of one motion event."""
    start_frame: int
    end_frame: int | None = None
    detected_objects: list[str] = field(default_factory=list)  # Each event gets its own list.

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
    """Remember motion across frames until an event is finished."""

    def __init__(self, max_gap_frames: int = 0) -> None:
        """Create a tracker that can tolerate short gaps without motion."""
        self.current_event: MotionEvent | None = None  # The event still in progress.
        self.max_gap_frames = max_gap_frames  # Allowed gap without motion.
        self.gap_frames = 0  # Current gap without motion.
        self.last_motion_frame: int | None = None  # Last frame that had motion.

    def update(self, frame_number: int, has_motion: bool) -> MotionEvent | None:
        """Update the tracker with the motion result of one video frame."""
        # Called once for each processed frame.
        if has_motion:
            self.gap_frames = 0  # Motion ends the current gap.
            self.last_motion_frame = frame_number
            if self.current_event is None:  # Start a new event if needed.
                self.current_event = MotionEvent(start_frame=frame_number)
            return None

        if self.current_event is None:  # No active event to close.
            return None

        # Keep the same event through short gaps, such as 1 or 2 quiet frames.
        self.gap_frames += 1
        if self.gap_frames <= self.max_gap_frames:
            return None

        # The quiet gap is too long; end at the last frame with real motion.
        self.current_event.close(end_frame=self.last_motion_frame)

        # Save the event before clearing the tracker's current state.
        completed_event = self.current_event
        self.current_event = None
        self.gap_frames = 0
        self.last_motion_frame = None
        return completed_event


    def finish(self) -> MotionEvent | None:
        """Close and return the active event when video processing finishes."""
        # Example: current_event = MotionEvent(start_frame=100, end_frame=None)
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
