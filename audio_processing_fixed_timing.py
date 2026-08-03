from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger("vocal-stutter-transfer.audio")

TARGET_SR = 22050
MAX_DURATION_SECONDS = 300.0

# Tuning for tighter, more connected stutters.
FIRST_REPEAT_OFFSET_MS = 12.0
MIN_INTERVAL_MS = 42.0
MAX_INTERVAL_MS = 62.0
DEFAULT_INTERVAL_MS = 52.0
MIN_SLICE_MS = 52.0
MAX_SLICE_MS = 82.0

# Keep repeat volume nearly even to avoid obvious pumping.
REPEAT_DECAY_DB = [0.0, -0.25, -0.5, -0.75, -1.0, -1.25]


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

    y = np.nan_to_num(y).astype(np.float32)
    return y, sr


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-12))


def _detect_onsets(y: np.ndarray, sr: int) -> np.ndarray:
    hop = 256
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=hop,
        aggregate=np.median,
    )

    frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop,
        backtrack=True,
        units="frames",
    )

    return librosa.frames_to_time(frames, sr=sr, hop_length=hop)


def _detect_stutter_events(reference: np.ndarray, sr: int) -> list[dict]:
    onset_times = _detect_onsets(reference, sr)

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

        raw_interval_ms = float(np.median([a, b]) * 1000.0)
        interval_ms = float(
            np.clip(raw_interval_ms, MIN_INTERVAL_MS, MAX_INTERVAL_MS)
        )

        slice_ms = float(
            np.clip(interval_ms * 1.12, MIN_SLICE_MS, MAX_SLICE_MS)
        )

        repeats = 3
        if i + 2 < len(intervals):
            c = float(intervals[i + 2])
            if (
                0.025 <= c <= 0.22
                and abs(c - (interval_ms / 1000.0))
                / max(c, interval_ms / 1000.0)
                <= 0.35
            ):
                repeats = 4

        events.append(
            {
                "time": float(onset_times[i]),
                "interval_ms": interval_ms,
                "slice_ms": slice_ms,
                "repeats": repeats,
            }
        )

    # Merge near-duplicate detections.
    merged: list[dict] = []
    for event in events:
        if merged and event["time"] - merged[-1]["time"] < 0.16:
            if event["repeats"] > merged[-1]["repeats"]:
                merged[-1] = event
        else:
            merged.append(event)

    return merged



def _ensure_full_reference_coverage(
    events: list[dict],
    reference: np.ndarray,
    sr: int,
    max_gap_seconds: float = 4.0,
) -> list[dict]:
    """
    Preserve detected stutters, then add conservative events in long empty
    sections so the effect can continue through the whole reference,
    including its last 30 seconds.
    """
    duration = len(reference) / sr
    if duration <= 0:
        return events

    onset_times = _detect_onsets(reference, sr)
    result = [dict(event) for event in events]

    # Always consider the entire reference, not only the region where strong
    # repeated onsets were detected.
    anchors = [0.0]
    anchors.extend(sorted(float(e["time"]) for e in result))
    anchors.append(duration)

    additions: list[dict] = []

    for left, right in zip(anchors[:-1], anchors[1:]):
        gap = right - left
        if gap <= max_gap_seconds:
            continue

        # Fill long gaps at a restrained cadence. Snap each proposed time to
        # the closest actual onset when possible.
        count = max(1, int(gap // max_gap_seconds))
        proposed = np.linspace(left, right, count + 2)[1:-1]

        for t in proposed:
            chosen = float(t)
            if len(onset_times):
                nearby = onset_times[np.abs(onset_times - t) <= 0.45]
                if len(nearby):
                    chosen = float(nearby[np.argmin(np.abs(nearby - t))])

            additions.append(
                {
                    "time": float(np.clip(chosen, 0.05, max(0.05, duration - 0.05))),
                    "interval_ms": DEFAULT_INTERVAL_MS,
                    "slice_ms": 68.0,
                    "repeats": 3,
                }
            )

    # Explicitly guarantee coverage in the final 30 seconds.
    tail_start = max(0.0, duration - 30.0)
    tail_events = [float(e["time"]) for e in result + additions if float(e["time"]) >= tail_start]

    if duration > 10.0 and not tail_events:
        tail_candidates = onset_times[onset_times >= tail_start] if len(onset_times) else np.array([])
        if len(tail_candidates):
            step = max(1, len(tail_candidates) // 4)
            selected = tail_candidates[::step][:4]
        else:
            selected = np.linspace(
                max(tail_start + 0.5, 0.5),
                max(tail_start + 0.6, duration - 0.5),
                num=4,
            )

        for t in selected:
            additions.append(
                {
                    "time": float(np.clip(t, 0.05, max(0.05, duration - 0.05))),
                    "interval_ms": DEFAULT_INTERVAL_MS,
                    "slice_ms": 68.0,
                    "repeats": 3,
                }
            )

    combined = result + additions
    combined.sort(key=lambda event: float(event["time"]))

    # Remove near-duplicates while retaining full-song coverage.
    deduped: list[dict] = []
    for event in combined:
        if deduped and float(event["time"]) - float(deduped[-1]["time"]) < 0.20:
            continue
        deduped.append(event)

    return deduped


def _apply_density(events: list[dict], density: float) -> list[dict]:
    density = float(np.clip(density, 0.2, 1.0))

    if density >= 0.999 or len(events) <= 1:
        return events

    keep = max(1, round(len(events) * density))
    indices = np.linspace(0, len(events) - 1, keep).round().astype(int)
    return [events[int(i)] for i in np.unique(indices)]


def _equal_power_fades(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 1:
        ones = np.ones(1, dtype=np.float32)
        return ones, ones

    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return (
        np.sin(x * np.pi / 2.0),
        np.cos(x * np.pi / 2.0),
    )


def _soft_clip(y: np.ndarray, threshold: float = 0.995) -> np.ndarray:
    """
    Transparent safety clipping without changing the level of the whole file.
    Samples below the threshold are untouched, so the dry vocal cannot pump.
    """
    y = y.astype(np.float32, copy=True)
    abs_y = np.abs(y)
    mask = abs_y > threshold

    if np.any(mask):
        excess = abs_y[mask] - threshold
        headroom = max(1e-6, 1.0 - threshold)
        y[mask] = np.sign(y[mask]) * (
            threshold + headroom * np.tanh(excess / headroom)
        )

    return np.clip(y, -0.999, 0.999).astype(np.float32)


def _render(
    dry: np.ndarray,
    sr: int,
    events: list[dict],
    reference_duration: float,
    wet_db: float,
    strength: float,
    crossfade_ms: float,
) -> np.ndarray:
    """
    Render stutters as a controlled crossfade/replacement instead of simply
    adding them on top of the dry vocal. This keeps perceived volume stable.
    """
    wet = np.zeros_like(dry, dtype=np.float32)
    wet_weight = np.zeros_like(dry, dtype=np.float32)

    dry_duration = len(dry) / sr

    wet_db = float(np.clip(wet_db, -18.0, 0.0))
    strength = float(np.clip(strength, 0.25, 1.5))
    crossfade_ms = float(np.clip(crossfade_ms, 2.0, 16.0))

    # Convert the wet control to a blend amount. At -3 dB this is about 0.71.
    blend = float(np.clip(10.0 ** (wet_db / 20.0), 0.10, 1.0))
    crossfade_samples = max(1, int(sr * crossfade_ms / 1000.0))
    first_offset_samples = max(1, int(sr * FIRST_REPEAT_OFFSET_MS / 1000.0))

    for event in events:
        event_time = (float(event["time"]) / max(reference_duration, 1e-6)) * dry_duration

        interval_ms = float(
            np.clip(
                event.get("interval_ms", DEFAULT_INTERVAL_MS),
                MIN_INTERVAL_MS,
                MAX_INTERVAL_MS,
            )
        )
        slice_ms = float(
            np.clip(
                event.get("slice_ms", 64.0),
                MIN_SLICE_MS,
                MAX_SLICE_MS,
            )
        )
        repeats = int(np.clip(event.get("repeats", 3), 2, 6))

        slice_samples = max(32, int(sr * slice_ms / 1000.0))
        center = int(event_time * sr)

        source_end = min(len(dry), center + int(sr*0.015))
        source_start = max(0, source_end - slice_samples)
        snippet = dry[source_start:source_end].copy()

        if len(snippet) < 32:
            continue

        snippet -= float(np.mean(snippet))

        # Match every repeated fragment to the source fragment's own RMS.
        # This prevents one repeat from suddenly being much louder than another.
        snippet_rms = _rms(snippet)
        if snippet_rms > 1e-9:
            local_target = max(_rms(dry[source_start:source_end]), 1e-9)
            snippet *= float(np.clip(local_target / snippet_rms, 0.75, 1.25))

        fade_len = min(crossfade_samples, max(1, len(snippet) // 8))
        fade_in, fade_out = _equal_power_fades(fade_len)

        envelope = np.ones(len(snippet), dtype=np.float32)
        envelope[:fade_len] *= fade_in
        envelope[-fade_len:] *= fade_out

        spacing = max(
            int(sr * interval_ms / 1000.0),
            int(len(snippet) * 0.82),
        )
        first_repeat = source_end + int(sr*0.012)

        for repeat_index in range(repeats):
            start = first_repeat + repeat_index * spacing
            end = min(len(dry), start + len(snippet))

            if start >= len(dry) or end <= start:
                break

            decay_db = REPEAT_DECAY_DB[
                min(repeat_index, len(REPEAT_DECAY_DB) - 1)
            ]
            decay_gain = 10.0 ** (decay_db / 20.0)

            length = end - start
            event_env = envelope[:length]
            repeated = snippet[:length] * decay_gain * strength

            # Accumulate a weighted stutter signal and a matching blend mask.
            wet[start:end] += repeated * event_env
            wet_weight[start:end] += event_env

    # Average overlapping repeats instead of summing them louder.
    active = wet_weight > 1e-6
    wet[active] /= wet_weight[active]

    # Cap the mask at 1 so overlapping events cannot raise the output level.
    mask = np.clip(wet_weight, 0.0, 1.0) * blend

    # Crossfade from dry to wet only where the stutter exists.
    output = dry * (1.0 - mask) + wet * mask

    return _soft_clip(output)

def process_stutter_transfer(
    reference_path: Path,
    dry_path: Path,
    output_path: Path,
    wet_db: float = -3.0,
    density: float = 1.0,
    strength: float = 1.1,
    crossfade_ms: float = 6.0,
    prompt: str = "",
) -> dict:
    del prompt

    with tempfile.TemporaryDirectory(prefix="vst_convert_") as temp_dir:
        temp_path = Path(temp_dir)
        reference_wav = temp_path / "reference.wav"
        dry_wav = temp_path / "dry.wav"

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
        events = _ensure_full_reference_coverage(events, reference, sr)
        events = _apply_density(events, density)

        if not events:
            logger.warning(
                "No strong repeated stutters detected; using tight fallback events"
            )

            fallback_count = min(
                12,
                max(4, int(reference_duration // 5) + 1),
            )
            fallback_times = np.linspace(
                min(0.5, max(0.05, reference_duration * 0.05)),
                max(0.6, reference_duration * 0.95),
                num=fallback_count,
            )

            events = [
                {
                    "time": float(t),
                    "interval_ms": DEFAULT_INTERVAL_MS,
                    "slice_ms": 64.0,
                    "repeats": 4,
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

        sf.write(
            output_path,
            processed,
            sr,
            subtype="PCM_24",
        )

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("The output WAV was not created correctly.")

    return {
        "event_count": len(events),
        "reference_duration": round(reference_duration, 3),
        "dry_duration": round(dry_duration, 3),
        "first_repeat_offset_ms": FIRST_REPEAT_OFFSET_MS,
        "interval_range_ms": [MIN_INTERVAL_MS, MAX_INTERVAL_MS],
    }
