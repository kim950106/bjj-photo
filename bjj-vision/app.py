from pathlib import Path
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bjj.models import AnalysisConfig
from bjj.analyzer import BJJAnalyzerV0
from bjj.nvidia_analyzer import analyze_video_with_nvidia
from bjj.rules import BELT_MATCH_MINUTES, DEFAULT_GYM_ROLL_MINUTES, POINTS, event_points

app = FastAPI(title="BJJ Vision", version="0.2.0")
analyzer = BJJAnalyzerV0()

STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")

@app.get("/")
def root():
    return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health():
    return {"ok": True, "engine": "nvidia-cosmos-ready"}

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


def _validate_video(video: UploadFile) -> str:
    if not video.filename:
        raise HTTPException(400, "파일명이 없습니다.")
    suffix = Path(video.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".avi", ".webm"}:
        raise HTTPException(400, "지원 영상 형식: mp4, mov, m4v, avi, webm")
    return suffix


@app.post("/api/analyze-nvidia")
async def analyze_nvidia(video: UploadFile = File(...)):
    suffix = _validate_video(video)
    data = await video.read()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(413, "500MB 이하 영상을 사용해주세요.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        result = await analyze_video_with_nvidia(tmp_path)
        score = {"A": 0, "B": 0}
        scoring_events = []
        for event in result.get("events", []):
            actor = event.get("actor")
            event_type = event.get("type", "")
            stable = float(event.get("stable_seconds", 0) or 0)
            pts = event_points(event_type, stable)
            if actor in score and pts:
                score[actor] += pts
                scoring_events.append({**event, "points": pts})
        result["score"] = score
        result["scoring_events"] = scoring_events
        result["warning"] = "AI 판정은 베타입니다. 낮은 신뢰도 이벤트는 사용자 확인이 필요합니다."
        return result
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(400, f"NVIDIA 영상 분석 실패: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/analyze")
async def analyze(
    video: UploadFile = File(...),
    my_belt: str = Form("blue"),
    opponent_belt: str = Form("blue"),
    mode: str = Form("training"),
    round_minutes: float = Form(5),
    my_start_side: str = Form("left"),
):
    suffix = _validate_video(video)
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

    data = await video.read()
    if len(data) > 500 * 1024 * 1024:
        raise HTTPException(413, "500MB 이하 영상을 사용해주세요.")

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
