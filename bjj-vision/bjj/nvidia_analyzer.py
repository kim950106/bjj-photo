from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import cv2
import httpx

NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
COSMOS_MODEL = "nvidia/cosmos3-nano-reasoner"

SYSTEM_PROMPT = """You analyze Brazilian Jiu-Jitsu sparring video. Be conservative. Do not invent events. Return JSON only. Athlete A is the athlete initially on the left; athlete B is initially on the right. Track identity by clothing, belt, patches, body shape and continuity. Distinguish position state from scoring event. Back control requires leg-hook/control evidence; simply being behind is not back control. Sweep must originate from guard/half guard. Guard pass must originate from guard/half guard. Full-score control normally requires about 3 seconds of stabilization."""

USER_PROMPT = """Analyze this BJJ video chunk. Return ONLY valid JSON with this schema:
{
  "summary": "short Korean summary",
  "athlete_a_description": "visible identity cues",
  "athlete_b_description": "visible identity cues",
  "events": [
    {
      "start": 0.0,
      "end": 3.0,
      "actor": "A|B|unknown",
      "type": "standing|guard_pull|takedown|guard|half_guard|sweep|reversal|guard_pass|side_control|north_south|knee_on_belly|mount|back_attempt|back_control|submission_attempt|tap|escape|scramble|unknown",
      "confidence": 0.0,
      "stable_seconds": 0.0,
      "evidence": "brief visible evidence"
    }
  ]
}
Times are seconds relative to this chunk. Prefer unknown over guessing. For back_control, only use it when leg hooks/control are visibly supported. For tap, only use it when a tap/stop signal is actually visible; audio is handled separately."""


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError("NVIDIA 응답에서 JSON을 찾지 못했습니다.")
        return json.loads(m.group(0))


def _make_chunk(source: Path, start_sec: float, end_sec: float, target_fps: float = 2.0, max_width: int = 640) -> Path:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError("영상을 열 수 없습니다.")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    scale = min(1.0, max_width / max(1, width))
    out_w = max(2, int(width * scale) // 2 * 2)
    out_h = max(2, int(height * scale) // 2 * 2)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.close()
    out_path = Path(tmp.name)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        out_path.unlink(missing_ok=True)
        raise ValueError("분석용 mp4 생성에 실패했습니다.")

    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    next_sample = start_sec
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if current > end_sec:
            break
        if current + 1e-3 >= next_sample:
            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            next_sample += 1.0 / target_fps

    writer.release()
    cap.release()
    return out_path


async def _call_cosmos(chunk_path: Path) -> dict[str, Any]:
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY가 설정되지 않았습니다.")

    video_b64 = base64.b64encode(chunk_path.read_bytes()).decode("ascii")
    payload = {
        "model": COSMOS_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{video_b64}"}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 4096,
        "stream": False,
        "media_io_kwargs": {"video": {"fps": 2.0}},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    async with httpx.AsyncClient(timeout=180.0) as client:
        r = await client.post(NVIDIA_API_URL, headers=headers, json=payload)
        r.raise_for_status()
        body = r.json()
    text = body["choices"][0]["message"]["content"]
    return _extract_json(text)


async def analyze_video_with_nvidia(source: Path, chunk_seconds: int = 30, max_minutes: int = 10) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frames / fps if fps else 0
    cap.release()
    if duration <= 0:
        raise ValueError("영상 길이를 읽지 못했습니다.")

    duration = min(duration, max_minutes * 60)
    all_events: list[dict[str, Any]] = []
    summaries: list[str] = []
    identity = {"A": "", "B": ""}

    start = 0.0
    while start < duration:
        end = min(duration, start + chunk_seconds)
        chunk = _make_chunk(source, start, end)
        try:
            result = await _call_cosmos(chunk)
        finally:
            chunk.unlink(missing_ok=True)

        if result.get("summary"):
            summaries.append(result["summary"])
        if not identity["A"] and result.get("athlete_a_description"):
            identity["A"] = result["athlete_a_description"]
        if not identity["B"] and result.get("athlete_b_description"):
            identity["B"] = result["athlete_b_description"]

        for event in result.get("events", []):
            try:
                event["start"] = round(float(event.get("start", 0)) + start, 2)
                event["end"] = round(float(event.get("end", event["start"] - start)) + start, 2)
                event["confidence"] = max(0.0, min(1.0, float(event.get("confidence", 0))))
                event["stable_seconds"] = max(0.0, float(event.get("stable_seconds", 0)))
                all_events.append(event)
            except (TypeError, ValueError):
                continue
        start = end

    return {
        "engine": COSMOS_MODEL,
        "duration_sec": round(duration, 2),
        "athletes": identity,
        "summary": " ".join(summaries),
        "events": all_events,
    }
