from pydantic import BaseModel, Field
from typing import Optional, List

class AnalysisConfig(BaseModel):
    my_belt: str = "blue"
    opponent_belt: str = "blue"
    mode: str = "training"
    round_minutes: float = Field(default=5, ge=1, le=30)
    my_start_side: str = "left"

class DetectedPlayer(BaseModel):
    player_id: str
    initial_side: str
    gi_color: Optional[str] = None
    belt_color: Optional[str] = None
    marker_features: List[str] = []

class TimelineEvent(BaseModel):
    timestamp_sec: float
    event_type: str
    athlete: Optional[str] = None
    position: Optional[str] = None
    points: int = 0
    confidence: float = Field(default=0.0, ge=0, le=1)
    note: str = ""

class AnalysisResult(BaseModel):
    filename: str
    duration_sec: float
    sampled_frames: int
    config: AnalysisConfig
    players: List[DetectedPlayer]
    timeline: List[TimelineEvent]
    score_a: int = 0
    score_b: int = 0
    engine_stage: str
    message: str
