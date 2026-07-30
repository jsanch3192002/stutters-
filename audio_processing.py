from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger("vocal-stutter-transfer.audio")

TARGET_SR = 48000
MAX_DURATION_SECONDS = 300.0


def _run_ffmpeg(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-c:a", "pcm_f32le", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "FFmpeg could not decode the audio file.")


def _load(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("An uploaded file contains no readable audio.")
    duration = y.size / float(sr)
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(f"Audio is {duration:.1f}s long. Maximum is {int(MAX_DURATION_SECONDS)}s.")
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, copy=False)
    return y, int(sr)


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _rms_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20.0 * np.log10(max(rms, 1e-9))


def _detect_reference_events(reference: np.ndarray, sr: int) -> list[dict[str, float | int]]:
    hop = 256
    env = librosa.onset.onset_strength(y=reference, sr=sr, hop_length=hop, aggregate=np.median)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="frames", backtrack=False,
        delta=0.08, wait=1,
    )
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
    if len(times) < 3:
        return []

    intervals = np.diff(times)
    events: list[dict[str, float | int]] = []
    for i in range(len(intervals) - 1):
        a, b = float(intervals[i]), float(intervals[i + 1])
        if not (0.035 <= a <= 0.22 and 0.035 <= b <= 0.22):
            continue
        similarity = abs(a - b) / max(a, b)
        if similarity > 0.28:
            continue
        interval = float(np.median([a, b]))
        repeats = 3
        if i + 2 < len(intervals):
            c = float(intervals[i + 2])
            if 0.035 <= c <= 0.22 and abs(c - interval) / max(c, interval) <= 0.28:
                repeats = 4
        events.append({
            "time": float(times[i]),
            "interval_ms": interval * 1000.0,
            "slice_ms": float(np.clip(interval * 1000.0 * 0.92, 62.0, 96.0)),
            "repeats": repeats,
        })

    merged: list[dict[str, float | int]] = []
    for event in events:
        if merged and float(event["time"]) - float(merged[-1]["time"]) < 0.16:
            if int(event["repeats"]) > int(merged[-1]["repeats"]):
                merged[-1] = event
        else:
            merged.append(event)
    return merged[:100]


def _apply_density(events: list[dict[str, float | int]], density: float) -> list[dict[str, float | int]]:
    density = float(np.clip(density, 0.15, 1.0))
    if density >= 0.999 or len(events) < 2:
        return events
    count = max(1, round(len(events) * density))
    idx = np.unique(np.linspace(0, len(events) - 1, count).round().astype(int))
    return [events[int(i)] for i in idx]


def _equal_power_fades(length: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, max(1, length), dtype=np.float32)
    return np.sin(x * np.pi / 2.0), np.cos(x * np.pi / 2.0)


def _transparent_limiter(y: np.ndarray, ceiling: float = 0.988) -> np.ndarray:
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= ceiling:
        return y.astype(np.float32, copy=False)
    # Only reduce enough to prevent clipping. This preserves the dry vocal tone.
    return (y * (ceiling / peak)).astype(np.float32)


def _render(
    dry: np.ndarray,
    sr: int,
    events: list[dict[str, float | int]],
    reference_duration: float,
    wet_db: float,
    strength: float,
    crossfade_ms: float,
) -> np.ndarray:
    wet = np.zeros_like(dry, dtype=np.float32)
    dry_duration = dry.size / float(sr)
    scale = dry_duration / max(reference_duration, 1e-6)

    wet_gain = _db_to_gain(float(np.clip(wet_db, -18.0, 3.0)))
    strength = float(np.clip(strength, 0.25, 1.6))
    crossfade_samples = max(8, int(sr * float(np.clip(crossfade_ms, 2.0, 20.0)) / 1000.0))
    decay_db = (0.0, -0.5, -1.0, -1.5, -2.0, -2.5)

    for event in events:
        center = int(float(event["time"]) * scale * sr)
        slice_samples = max(64, int(float(event["slice_ms"]) * sr / 1000.0))
        interval_samples = max(slice_samples, int(float(event["interval_ms"]) * sr / 1000.0))
        repeats = int(event["repeats"])

        source_end = min(dry.size, max(slice_samples, center))
        source_start = max(0, source_end - slice_samples)
        snippet = dry[source_start:source_end].copy()
        if snippet.size < 64:
            continue

        fade_len = min(crossfade_samples, max(8, snippet.size // 6))
        fade_in, fade_out = _equal_power_fades(fade_len)
        snippet[:fade_len] *= fade_in
        snippet[-fade_len:] *= fade_out

        # Keep the repeat at the same perceived level as its source, with a modest maximum boost.
        local_start = max(0, source_start - slice_samples)
        local_end = min(dry.size, source_end + slice_samples)
        local_rms = _rms_db(dry[local_start:local_end])
        snippet_rms = _rms_db(snippet)
        match_db = float(np.clip(local_rms - snippet_rms, -1.5, 3.5))
        snippet *= _db_to_gain(match_db)

        first_start = source_end + int(0.004 * sr)
        for r in range(repeats):
            start = first_start + r * interval_samples
            if start >= wet.size:
                break
            end = min(wet.size, start + snippet.size)
            gain = _db_to_gain(decay_db[min(r, len(decay_db) - 1)]) * strength
            wet[start:end] += snippet[: end - start] * gain

    # Do not normalize the dry vocal. Control only the added stutter bus.
    wet_peak = float(np.max(np.abs(wet))) if wet.size else 0.0
    if wet_peak > 1.25:
        wet *= 1.25 / wet_peak
    return _transparent_limiter(dry + wet * wet_gain)


def process_stutter_transfer(
    reference_path: Path,
    dry_path: Path,
    output_path: Path,
    wet_db: float = -3.0,
    density: float = 1.0,
    strength: float = 1.15,
    crossfade_ms: float = 8.0,
    prompt: str = "",
) -> dict[str, float | int]:
    del prompt
    with tempfile.TemporaryDirectory(prefix="vst_audio_") as temp:
        temp_dir = Path(temp)
        reference_wav = temp_dir / "reference.wav"
        dry_wav = temp_dir / "dry.wav"
        _run_ffmpeg(reference_path, reference_wav)
        _run_ffmpeg(dry_path, dry_wav)
        reference, sr = _load(reference_wav)
        dry, dry_sr = _load(dry_wav)
        if sr != dry_sr:
            raise ValueError("Internal sample-rate mismatch.")

        reference_duration = reference.size / float(sr)
        dry_duration = dry.size / float(sr)
        events = _apply_density(_detect_reference_events(reference, sr), density)
        if not events:
            logger.warning("No strong repeated reference events detected; using safe fallback timing.")
            count = max(1, min(4, int(reference_duration // 8) + 1))
            events = [{
                "time": float(t), "interval_ms": 72.0, "slice_ms": 72.0, "repeats": 4
            } for t in np.linspace(reference_duration * 0.25, reference_duration * 0.75, count)]

        processed = _render(
            dry=dry, sr=sr, events=events, reference_duration=reference_duration,
            wet_db=wet_db, strength=strength, crossfade_ms=crossfade_ms,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, sr, subtype="PCM_24")
        if not output_path.is_file() or output_path.stat().st_size < 44:
            raise RuntimeError("Failed to create a valid output WAV file.")

    return {
        "event_count": len(events),
        "reference_duration": round(reference_duration, 3),
        "dry_duration": round(dry_duration, 3),
    }
