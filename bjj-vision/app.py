from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bjj.models import AnalysisConfig
from bjj.analyzer import BJJAnalyzerV0
from bjj.rules import BELT_MATCH_MINUTES, DEFAULT_GYM_ROLL_MINUTES, POINTS

app = FastAPI(title="BJJ Vision V0", version="0.1.0")
analyzer = BJJAnalyzerV0()

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")

@app.get("/api/rules")
def rules():
    return {
        "belt_match_minutes": BELT_MATCH_MINUTES,
        "default_gym_roll_minutes": DEFAULT_GYM_ROLL_MINUTES,
        "points": POINTS,
        "stabilization_seconds": 3,
        "player_identity_priority": [
            "initial_left_right",
            "gi_or_rashguard_color",
            "belt_color",
            "gi_patches_or_marks",
            "body_shape_and_motion",
        ],
        "tap_signals": [
            "visual_hand_tap",
            "visual_foot_tap",
            "verbal_tap_or_stop",
            "pain_scream_context",
            "tap_impact_sound_context",
        ],
    }

@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    my_belt: str = Form("blue"),
    opponent_belt: str = Form("blue"),
    mode: str = Form("training"),
    round_minutes: float = Form(5),
    my_start_side: str = Form("left"),
):
    if not video.filename:
        raise HTTPException(400, "파일명이 없습니다.")

    suffix = Path(video.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm"}:
        raise HTTPException(400, "지원 영상 형식: mp4, mov, m4v, avi, webm")

    try:
        config = AnalysisConfig(
            my_belt=my_belt,
            opponent_belt=opponent_belt,
            mode=mode,
            round_minutes=round_minutes,
            my_start_side=my_start_side,
        )
    except Exception as e:
        raise HTTPException(400, f"설정 오류: {e}")

    max_bytes = 500 * 1024 * 1024
    data = await video.read()
    if len(data) > max_bytes:
        raise HTTPException(413, "V0에서는 500MB 이하 영상을 사용해주세요.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        result = analyzer.analyze(tmp_path, video.filename, config)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(400, f"영상 분석 준비 실패: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)
