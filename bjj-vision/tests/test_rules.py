from bjj.rules import belt_match_seconds, event_points, ScoreEvent, score_events

def test_belt_times():
    assert belt_match_seconds("white") == 300
    assert belt_match_seconds("blue") == 360
    assert belt_match_seconds("black") == 600

def test_stabilization():
    assert event_points("mount", 2.9) == 0
    assert event_points("mount", 3.0) == 4
    assert event_points("guard_pass", 3.0) == 3

def test_score():
    events = [
        ScoreEvent("takedown", "A", 12, 3),
        ScoreEvent("guard_pass", "A", 44, 3),
        ScoreEvent("mount", "B", 70, 3),
    ]
    assert score_events(events) == {"A": 5, "B": 4}
