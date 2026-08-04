from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

logger = logging.getLogger("vocal-stutter-transfer.audio")

TARGET_SR = 22050
MAX_DURATION_SECONDS = 300.0
FIRST_REPEAT_OFFSET_MS = 18.0
MIN_INTERVAL_MS = 95.0
DEFAULT_INTERVAL_MS = 105.0
MAX_INTERVAL_MS = 118.0
MIN_SLICE_MS = 72.0
DEFAULT_SLICE_MS = 84.0
MAX_SLICE_MS = 100.0
DEFAULT_REPEATS = 4
REPEAT_DECAY_DB = (0.0, -0.35, -0.7, -1.05, -1.4, -1.75)


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-12))


def _convert_to_wav(source: Path, destination: Path) -> None:
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le", str(destination),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise ValueError(completed.stderr.strip() or "FFmpeg could not decode the audio.")


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = np.nan_to_num(np.asarray(y, dtype=np.float32))
    if y.size == 0:
        raise ValueError("One uploaded file contains no readable audio.")
    duration = len(y) / float(sr)
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(f"Audio is too long ({duration:.1f}s). Maximum supported length is {int(MAX_DURATION_SECONDS)} seconds.")
    return y, int(sr)


def _moving_average(x: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return x.astype(np.float32, copy=True)
    kernel = np.ones(size, dtype=np.float32) / float(size)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _detect_word_like_regions(y: np.ndarray, sr: int) -> list[tuple[float, float]]:
    frame_ms, hop_ms = 20.0, 10.0
    frame = max(64, int(sr * frame_ms / 1000.0))
    hop = max(32, int(sr * hop_ms / 1000.0))
    if len(y) < frame:
        return []

    energies, positions = [], []
    for start in range(0, len(y) - frame + 1, hop):
        energies.append(_rms(y[start:start + frame]))
        positions.append(start)

    env = _moving_average(np.asarray(energies, dtype=np.float32), 5)
    if env.size < 4:
        return []

    floor = float(np.percentile(env, 20))
    speech = float(np.percentile(env, 75))
    threshold = floor + max(1e-6, speech - floor) * 0.22
    active = env >= threshold

    max_hole = max(1, int(55.0 / hop_ms))
    i = 0
    while i < len(active):
        if active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and not active[j]:
            j += 1
        if i > 0 and j < len(active) and (j - i) <= max_hole:
            active[i:j] = True
        i = j

    min_region = max(1, int(70.0 / hop_ms))
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        if (j - i) < min_region:
            active[i:j] = False
        i = j

    raw = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        raw.append((positions[i], min(len(y), positions[min(j - 1, len(positions) - 1)] + frame)))
        i = j

    regions = []
    for start_sample, end_sample in raw:
        duration = (end_sample - start_sample) / float(sr)
        if duration <= 0.75:
            regions.append((start_sample / sr, end_sample / sr))
            continue

        start_frame = max(0, start_sample // hop)
        end_frame = min(len(env), int(np.ceil(end_sample / hop)))
        local = env[start_frame:end_frame]
        if local.size < 8:
            regions.append((start_sample / sr, end_sample / sr))
            continue

        valley_threshold = float(np.percentile(local, 35))
        min_word_frames = max(1, int(110.0 / hop_ms))
        splits, last = [], 0
        for idx in range(2, len(local) - 2):
            if local[idx] <= valley_threshold and local[idx] <= local[idx-1] and local[idx] <= local[idx+1] and (idx-last) >= min_word_frames and (len(local)-idx) >= min_word_frames:
                splits.append(idx)
                last = idx

        bounds = [0] + splits + [len(local)]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if (b - a) < min_word_frames:
                continue
            seg_start = start_sample + a * hop
            seg_end = min(end_sample, start_sample + b * hop + frame)
            if (seg_end - seg_start) / float(sr) >= 0.09:
                regions.append((seg_start / sr, seg_end / sr))

    cleaned = []
    for start, end in sorted(regions):
        if end <= start:
            continue
        if cleaned and start < cleaned[-1][1]:
            start = cleaned[-1][1]
        if end - start >= 0.07:
            cleaned.append((start, end))
    logger.info("Detected %d word-like vocal regions", len(cleaned))
    return cleaned


def _apply_density(regions: list[tuple[float, float]], density: float) -> list[tuple[float, float]]:
    density = float(np.clip(density, 0.2, 1.0))
    if density >= 0.999 or len(regions) <= 1:
        return regions
    keep = max(1, round(len(regions) * density))
    indices = np.linspace(0, len(regions) - 1, keep).round().astype(int)
    return [regions[int(i)] for i in np.unique(indices)]


def _equal_power_fades(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 1:
        ones = np.ones(1, dtype=np.float32)
        return ones, ones
    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.sin(x*np.pi/2.0), np.cos(x*np.pi/2.0)


def _soft_clip(y: np.ndarray, ceiling: float = 0.995) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).copy()
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= ceiling:
        return y
    drive = max(1.0, peak / ceiling)
    limited = np.tanh(y * drive) / np.tanh(drive)
    p = float(np.max(np.abs(limited)))
    if p > ceiling:
        limited *= ceiling / p
    return limited.astype(np.float32)


def _render(dry: np.ndarray, sr: int, regions: list[tuple[float, float]], wet_db: float, strength: float, crossfade_ms: float) -> np.ndarray:
    wet_db = float(np.clip(wet_db, -18.0, 0.0))
    strength = float(np.clip(strength, 0.25, 1.5))
    crossfade_ms = float(np.clip(crossfade_ms, 3.0, 20.0))

    interval_samples = int(sr * DEFAULT_INTERVAL_MS / 1000.0)
    slice_samples = int(sr * DEFAULT_SLICE_MS / 1000.0)
    first_offset = int(sr * FIRST_REPEAT_OFFSET_MS / 1000.0)
    fade_samples = int(sr * crossfade_ms / 1000.0)

    output = np.pad(dry.astype(np.float32), (0, int(sr * 1.25)))
    wet = np.zeros_like(output, dtype=np.float32)

    for region_start_sec, region_end_sec in regions:
        region_start = max(0, int(region_start_sec * sr))
        region_end = min(len(dry), int(region_end_sec * sr))
        if region_end <= region_start:
            continue

        source_end = region_end
        source_start = max(region_start, source_end - slice_samples)
        snippet = dry[source_start:source_end].copy()
        if len(snippet) < max(32, int(sr * 0.03)):
            continue

        snippet -= float(np.mean(snippet))
        fade_len = min(fade_samples, max(1, len(snippet)//7))
        fi, fo = _equal_power_fades(fade_len)
        snippet[:fade_len] *= fi
        snippet[-fade_len:] *= fo

        first_repeat = region_end + first_offset
        spacing = max(interval_samples, int(len(snippet) * 1.15))

        for repeat_index in range(DEFAULT_REPEATS):
            start = first_repeat + repeat_index * spacing
            end = min(len(wet), start + len(snippet))
            if start >= len(wet) or end <= start:
                break
            gain = _db_to_amp(REPEAT_DECAY_DB[min(repeat_index, len(REPEAT_DECAY_DB)-1)])
            wet[start:end] += snippet[:end-start] * gain * strength

    dry_rms, wet_rms = _rms(dry), _rms(wet)
    if dry_rms > 1e-9 and wet_rms > 1e-9:
        wet *= float(np.clip((dry_rms * 0.68) / wet_rms, 0.20, 1.35))

    output += wet * _db_to_amp(wet_db)
    return _soft_clip(output)


def process_stutter_transfer(
    reference_path: Path,
    dry_path: Path,
    output_path: Path,
    wet_db: float = -3.0,
    density: float = 1.0,
    strength: float = 1.0,
    crossfade_ms: float = 8.0,
    prompt: str = "",
) -> dict[str, Any]:
    del prompt
    with tempfile.TemporaryDirectory(prefix="vst_light_") as temp_dir:
        temp = Path(temp_dir)
        reference_wav = temp / "reference.wav"
        dry_wav = temp / "dry.wav"

        _convert_to_wav(reference_path, reference_wav)
        _convert_to_wav(dry_path, dry_wav)
        reference, reference_sr = _load_audio(reference_wav)
        dry, dry_sr = _load_audio(dry_wav)
        if reference_sr != dry_sr:
            raise ValueError("Internal sample-rate mismatch.")

        regions = _detect_word_like_regions(dry, dry_sr)
        if not regions:
            raise ValueError("No clear vocal regions were detected. Use an isolated dry vocal with less background noise.")
        selected = _apply_density(regions, density)
        processed = _render(dry, dry_sr, selected, wet_db, strength, crossfade_ms)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, dry_sr, subtype="PCM_24")
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("The output WAV was not created correctly.")

    return {
        "event_count": len(selected),
        "reference_duration": round(len(reference)/reference_sr, 3),
        "dry_duration": round(len(dry)/dry_sr, 3),
        "first_repeat_offset_ms": FIRST_REPEAT_OFFSET_MS,
        "interval_ms": DEFAULT_INTERVAL_MS,
        "slice_ms": DEFAULT_SLICE_MS,
        "repeats": DEFAULT_REPEATS,
        "engine": "lightweight-word-like-v1",
    }
