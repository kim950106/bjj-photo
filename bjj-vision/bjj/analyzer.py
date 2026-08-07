from pathlib import Path
from .models import AnalysisResult, DetectedPlayer, TimelineEvent
from .video import inspect_video

class BJJAnalyzerV0:
    def analyze(self, path: Path, filename: str, config):
        duration, sample_times = inspect_video(path)

        my_player = DetectedPlayer(
            player_id="A",
            initial_side=config.my_start_side,
            marker_features=["initial-side-seed"],
        )
        opp_side = "right" if config.my_start_side == "left" else "left"
        opponent = DetectedPlayer(
            player_id="B",
            initial_side=opp_side,
            marker_features=["initial-side-seed"],
        )

        timeline = []
        if duration > 0:
            timeline.append(TimelineEvent(
                timestamp_sec=0,
                event_type="round_start",
                position="standing",
                confidence=1.0,
                note="V0 starts from standing unless corrected by the user.",
            ))

        return AnalysisResult(
            filename=filename,
            duration_sec=duration,
            sampled_frames=len(sample_times),
            config=config,
            players=[my_player, opponent],
            timeline=timeline,
            score_a=0,
            score_b=0,
            engine_stage="V0_PREPROCESSING_READY",
            message=(
                "영상 전처리와 경기 설정 파이프라인은 동작합니다. "
                "실제 Guard/Half/Side/Mount/Back 자동 판정 모델은 다음 단계에서 연결합니다."
            ),
        )
