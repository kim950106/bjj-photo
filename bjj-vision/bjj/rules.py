from dataclasses import dataclass

BELT_MATCH_MINUTES = {
    "white": 5,
    "blue": 6,
    "purple": 7,
    "brown": 8,
    "black": 10,
}

DEFAULT_GYM_ROLL_MINUTES = 5

POINTS = {
    "takedown": 2,
    "sweep": 2,
    "knee_on_belly": 2,
    "guard_pass": 3,
    "mount": 4,
    "back_control": 4,
}

STABILIZATION_SECONDS = {
    "takedown": 3,
    "sweep": 3,
    "knee_on_belly": 3,
    "guard_pass": 3,
    "mount": 3,
    "back_control": 3,
}

POSITIONS = [
    "standing", "guard", "half_guard", "side_control", "north_south",
    "knee_on_belly", "mount", "back_attempt", "back_control", "turtle", "scramble",
]

@dataclass
class ScoreEvent:
    event_type: str
    athlete: str
    start_sec: float
    stable_for_sec: float = 0
    note: str = ""

def belt_match_seconds(belt: str) -> int:
    key = belt.strip().lower()
    if key not in BELT_MATCH_MINUTES:
        raise ValueError(f"Unsupported belt: {belt}")
    return BELT_MATCH_MINUTES[key] * 60

def event_points(event_type: str, stable_for_sec: float) -> int:
    pts = POINTS.get(event_type, 0)
    required = STABILIZATION_SECONDS.get(event_type, 0)
    if required and stable_for_sec < required:
        return 0
    return pts

def score_events(events):
    score = {"A": 0, "B": 0}
    for e in events:
        if e.athlete in score:
            score[e.athlete] += event_points(e.event_type, e.stable_for_sec)
    return score
