from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger("vocal-stutter-transfer.audio")

TARGET_SR = 22050
MAX_DURATION_SECONDS = 300.0


def _convert_to_wav(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or "FFmpeg could not decode the audio."
        raise ValueError(message)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    if y.size == 0:
        raise ValueError("One of the uploaded files contains no readable audio.")

    duration = len(y) / sr
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Audio is too long ({duration:.1f}s). Maximum supported length is "
            f"{int(MAX_DURATION_SECONDS)} seconds."
        )

    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / max(1.0, peak)

    return y.astype(np.float32), sr


def _detect_stutter_events(reference: np.ndarray, sr: int) -> list[dict]:
    hop = 256
    onset_env = librosa.onset.onset_strength(
        y=reference,
        sr=sr,
        hop_length=hop,
        aggregate=np.median,
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop,
        backtrack=False,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop)

    if len(onset_times) < 3:
        return []

    intervals = np.diff(onset_times)
    events: list[dict] = []

    for i in range(len(intervals) - 1):
        a = float(intervals[i])
        b = float(intervals[i + 1])

        if not (0.025 <= a <= 0.22 and 0.025 <= b <= 0.22):
            continue

        similarity = abs(a - b) / max(a, b)
        if similarity > 0.35:
            continue

        event_time = float(onset_times[i])
        interval = float(np.median([a, b]))
        slice_ms = float(np.clip(interval * 1000.0 * 0.78, 35.0, 120.0))
        repeats = 3

        if i + 2 < len(intervals):
            c = float(intervals[i + 2])
            if 0.025 <= c <= 0.22 and abs(c - interval) / max(c, interval) <= 0.35:
                repeats = 4

        events.append(
            {
                "time": event_time,
                "interval_ms": interval * 1000.0,
                "slice_ms": slice_ms,
                "repeats": repeats,
            }
        )

    # Merge events that are very close together.
    merged: list[dict] = []
    for event in events:
        if merged and event["time"] - merged[-1]["time"] < 0.18:
            if event["repeats"] > merged[-1]["repeats"]:
                merged[-1] = event
        else:
            merged.append(event)

    return merged[:80]


def _apply_density(events: list[dict], density: float) -> list[dict]:
    density = float(np.clip(density, 0.2, 1.0))
    if density >= 0.999 or len(events) <= 1:
        return events

    keep = max(1, round(len(events) * density))
    indices = np.linspace(0, len(events) - 1, keep).round().astype(int)
    return [events[int(i)] for i in np.unique(indices)]


def _equal_power_fade(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 1:
        return np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32)

    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    fade_in = np.sin(x * np.pi / 2.0)
    fade_out = np.cos(x * np.pi / 2.0)
    return fade_in, fade_out


def _render(
    dry: np.ndarray,
    sr: int,
    events: list[dict],
    reference_duration: float,
    wet_db: float,
    strength: float,
    crossfade_ms: float,
) -> np.ndarray:
    output = dry.copy()
    wet = np.zeros_like(dry, dtype=np.float32)

    dry_duration = len(dry) / sr
    scale = dry_duration / max(reference_duration, 1e-6)
    wet_gain = 10.0 ** (float(np.clip(wet_db, -24.0, 0.0)) / 20.0)
    strength = float(np.clip(strength, 0.25, 1.5))
    crossfade_samples = max(1, int(sr * float(np.clip(crossfade_ms, 2.0, 30.0)) / 1000.0))

    for event in events:
        center_time = event["time"] * scale
        slice_ms = float(event["slice_ms"])
        interval_ms = float(event["interval_ms"])
        repeats = int(event["repeats"])

        slice_samples = max(16, int(sr * slice_ms / 1000.0))
        center = int(center_time * sr)
        source_end = min(len(dry), max(slice_samples, center))
        source_start = max(0, source_end - slice_samples)
        snippet = dry[source_start:source_end].copy()

        if len(snippet) < 16:
            continue

        fade_len = min(crossfade_samples, max(1, len(snippet) // 4))
        fade_in, fade_out = _equal_power_fade(fade_len)
        snippet[:fade_len] *= fade_in
        snippet[-fade_len:] *= fade_out

        spacing = max(len(snippet), int(sr * interval_ms / 1000.0))
        first_repeat = source_end + max(1, int(0.004 * sr))

        for repeat_index in range(repeats):
            start = first_repeat + repeat_index * spacing
            end = min(len(wet), start + len(snippet))
            if start >= len(wet) or end <= start:
                break

            gain_step = 10.0 ** ((-2.0 * repeat_index) / 20.0)
            wet[start:end] += snippet[: end - start] * gain_step * strength

    output += wet * wet_gain
    peak = float(np.max(np.abs(output)))
    if peak > 0.98:
        output = output * (0.98 / peak)

    return output.astype(np.float32)


def process_stutter_transfer(
    reference_path: Path,
    dry_path: Path,
    output_path: Path,
    wet_db: float = -8.0,
    density: float = 1.0,
    strength: float = 1.0,
    crossfade_ms: float = 8.0,
    prompt: str = "",
) -> dict:
    del prompt  # Reserved for future prompt-based controls.

    with tempfile.TemporaryDirectory(prefix="vst_convert_") as tmp:
        tmp_dir = Path(tmp)
        reference_wav = tmp_dir / "reference.wav"
        dry_wav = tmp_dir / "dry.wav"

        logger.info("Converting reference audio")
        _convert_to_wav(reference_path, reference_wav)

        logger.info("Converting dry vocal")
        _convert_to_wav(dry_path, dry_wav)

        logger.info("Loading audio")
        reference, sr = _load_audio(reference_wav)
        dry, dry_sr = _load_audio(dry_wav)

        if sr != dry_sr:
            raise ValueError("Internal sample-rate mismatch.")

        reference_duration = len(reference) / sr
        dry_duration = len(dry) / sr

        logger.info("Detecting reference stutters")
        events = _detect_stutter_events(reference, sr)
        events = _apply_density(events, density)

        if not events:
            logger.warning("No strong repeated stutters detected; using fallback events")
            fallback_times = np.linspace(
                max(0.5, reference_duration * 0.2),
                max(0.6, reference_duration * 0.8),
                num=min(4, max(1, int(reference_duration // 8) + 1)),
            )
            events = [
                {
                    "time": float(t),
                    "interval_ms": 70.0,
                    "slice_ms": 55.0,
                    "repeats": 3,
                }
                for t in fallback_times
            ]

        logger.info("Rendering %s stutter events", len(events))
        processed = _render(
            dry=dry,
            sr=sr,
            events=events,
            reference_duration=reference_duration,
            wet_db=wet_db,
            strength=strength,
            crossfade_ms=crossfade_ms,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, sr, subtype="PCM_16")

    return {
        "event_count": len(events),
        "reference_duration": round(reference_duration, 3),
        "dry_duration": round(dry_duration, 3),
    }
