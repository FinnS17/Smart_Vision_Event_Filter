from smart_vision_event_filter.events import MotionEvent, MotionEventTracker

def test_new_motion_event_is_active():
    """Check that a new motion event starts in the active state."""
    event = MotionEvent(start_frame=10)

    assert event.start_frame == 10
    assert event.end_frame is None
    assert event.is_active


def test_motion_event_can_be_closed():
    """Check that an event stores its end frame when it is closed."""
    event = MotionEvent(start_frame=10)
    event.close(end_frame=25)

    assert not event.is_active
    assert event.end_frame == 25


def test_tracker_creates_event_from_consecutive_motion_frames():
    """Check that consecutive motion frames keep one event active."""
    tracker = MotionEventTracker()

    assert tracker.update(frame_number=10, has_motion = True) is None
    assert tracker.update(frame_number=11, has_motion = True) is None


def test_tracker_starts_without_active_event():
    """Check that a new tracker does not contain an active event."""
    tracker = MotionEventTracker()
    assert tracker.current_event is None


def test_tracker_completes_event_when_motion_stops():
    """Check that the tracker returns an event after motion stops."""
    tracker = MotionEventTracker()

    result = tracker.update(frame_number= 10, has_motion=True)
    assert result is None

    result = tracker.update(frame_number= 11, has_motion=True)
    assert result is None

    completed_event = tracker.update(frame_number= 12, has_motion=False)
    assert completed_event is not None
    assert completed_event.start_frame == 10
    assert completed_event.end_frame == 11

    assert tracker.current_event is None


def test_tracker_can_create_multiple_events():
    """Check that one tracker can create separate events over time."""
    tracker = MotionEventTracker()
    tracker.update(frame_number=10, has_motion=True)
    tracker.update(frame_number=11, has_motion=True)
    first_event = tracker.update(frame_number=12, has_motion=False)
    assert first_event is not None
    assert first_event.start_frame == 10
    assert first_event.end_frame == 11

    result = tracker.update(frame_number=13, has_motion=False)
    assert result is None

    tracker.update(frame_number=20, has_motion=True)
    tracker.update(frame_number=21, has_motion=True)
    tracker.update(frame_number=22, has_motion=True)
    second_event = tracker.update(frame_number=23, has_motion=False)
    assert second_event is not None
    assert second_event.start_frame == 20
    assert second_event.end_frame == 22

    assert first_event is not second_event


def test_tracker_tolerates_short_motion_gap():
    """Check that a short gap without motion does not split an event."""
    tracker = MotionEventTracker(max_gap_frames=2)
    tracker.update(frame_number=10, has_motion=True)
    result = tracker.update(frame_number=11, has_motion=False)
    assert result is None
    result = tracker.update(frame_number=12, has_motion=False)
    assert result is None
    result = tracker.update(frame_number=13, has_motion= True)
    assert result is None
    assert tracker.current_event is not None
    assert tracker.current_event.start_frame == 10
    assert tracker.last_motion_frame == 13
    assert tracker.gap_frames == 0


def test_tracker_finishes_active_event():
    """Check that finish closes an event at the end of the video."""
    tracker = MotionEventTracker(max_gap_frames=5)
    tracker.update(frame_number=10, has_motion=True)
    tracker.update(frame_number=11, has_motion=True)
    result = tracker.finish()
    assert result is not None
    assert result.start_frame == 10
    assert result.end_frame == 11
    assert tracker.current_event is None
    
