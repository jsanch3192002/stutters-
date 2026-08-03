from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger("vocal-stutter-transfer.word-aligned")

TARGET_SR = 22050
MAX_DURATION_SECONDS = 300.0

# Word-aligned stutter timing
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny.en")
FIRST_REPEAT_OFFSET_MS = 14.0
DEFAULT_INTERVAL_MS = 105.0
MIN_INTERVAL_MS = 90.0
MAX_INTERVAL_MS = 125.0
DEFAULT_SLICE_MS = 82.0
MIN_SLICE_MS = 70.0
MAX_SLICE_MS = 100.0
DEFAULT_REPEATS = 4
REPEAT_DECAY_DB = (0.0, -0.4, -0.8, -1.2, -1.6, -2.0)

_model: WhisperModel | None = None
_model_lock = threading.Lock()


def _get_model() -> WhisperModel:
    """Load the speech model once per Render instance."""
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            logger.info("Loading faster-whisper model: %s", MODEL_SIZE)
            _model = WhisperModel(
                MODEL_SIZE,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(1, min(4, os.cpu_count() or 1)),
                num_workers=1,
            )

    return _model


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
        check=False,
    )

    if completed.returncode != 0:
        message = completed.stderr.strip() or "FFmpeg could not decode the audio."
        raise ValueError(message)


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, always_2d=False)

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    y = np.asarray(y, dtype=np.float32)
    y = np.nan_to_num(y)

    if y.size == 0:
        raise ValueError("One uploaded file contains no readable audio.")

    duration = len(y) / float(sr)
    if duration > MAX_DURATION_SECONDS:
        raise ValueError(
            f"Audio is too long ({duration:.1f}s). "
            f"Maximum supported length is {int(MAX_DURATION_SECONDS)} seconds."
        )

    return y, int(sr)


def _rms(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y), dtype=np.float64) + 1e-12))


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _equal_power_fades(length: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 1:
        ones = np.ones(1, dtype=np.float32)
        return ones, ones

    x = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return (
        np.sin(x * np.pi / 2.0).astype(np.float32),
        np.cos(x * np.pi / 2.0).astype(np.float32),
    )


def _soft_clip(y: np.ndarray, ceiling: float = 0.995) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).copy()
    peak = float(np.max(np.abs(y))) if y.size else 0.0

    if peak <= ceiling:
        return y

    drive = max(1.0, peak / ceiling)
    limited = np.tanh(y * drive) / np.tanh(drive)

    limited_peak = float(np.max(np.abs(limited)))
    if limited_peak > ceiling:
        limited *= ceiling / limited_peak

    return limited.astype(np.float32)


def _transcribe_word_ends(wav_path: Path) -> list[dict[str, Any]]:
    """
    Return true word-level timestamps from the dry vocal.

    faster-whisper returns each word with start/end times. We use the word end
    as the stutter anchor so repeats cannot begin before the spoken word ends.
    """
    model = _get_model()

    logger.info("Transcribing dry vocal with word timestamps")
    segments, info = model.transcribe(
        str(wav_path),
        language="en",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 120,
            "speech_pad_ms": 40,
        },
        condition_on_previous_text=False,
    )

    words: list[dict[str, Any]] = []

    for segment in segments:
        if not segment.words:
            continue

        for word in segment.words:
            text = (word.word or "").strip()
            if not text:
                continue

            start = float(word.start or 0.0)
            end = float(word.end or start)

            if end <= start:
                continue

            words.append(
                {
                    "word": text,
                    "start": start,
                    "end": end,
                    "probability": float(word.probability or 0.0),
                }
            )

    # Remove obvious duplicate timestamps.
    deduped: list[dict[str, Any]] = []
    for word in words:
        if deduped and abs(word["end"] - deduped[-1]["end"]) < 0.025:
            if word["probability"] > deduped[-1]["probability"]:
                deduped[-1] = word
        else:
            deduped.append(word)

    logger.info("Detected %d words", len(deduped))
    return deduped


def _estimate_style_from_reference(reference: np.ndarray, sr: int) -> dict[str, float | int]:
    """
    Use the reference only as a gentle style guide.

    This deliberately does not copy reference event positions. It estimates a
    rough fragment length and loudness while keeping word placement tied to the
    dry-vocal word timestamps.
    """
    duration = len(reference) / float(sr)
    ref_rms = _rms(reference)

    # Conservative defaults are more stable than trying to overfit a full mix.
    slice_ms = DEFAULT_SLICE_MS
    interval_ms = DEFAULT_INTERVAL_MS
    repeats = DEFAULT_REPEATS

    # Slightly adapt slice length from reference transient density.
    frame = max(64, int(sr * 0.02))
    hop = max(32, int(sr * 0.01))

    if len(reference) > frame:
        energies = []
        for start in range(0, len(reference) - frame, hop):
            segment = reference[start:start + frame]
            energies.append(_rms(segment))

        if energies:
            energies_arr = np.asarray(energies)
            changes = np.maximum(0.0, np.diff(energies_arr))
            if changes.size:
                threshold = float(np.percentile(changes, 85))
                transient_count = int(np.sum(changes >= threshold)) if threshold > 0 else 0
                density = transient_count / max(duration, 1e-6)

                if density > 8:
                    slice_ms = 74.0
                elif density < 3:
                    slice_ms = 90.0

    return {
        "slice_ms": float(np.clip(slice_ms, MIN_SLICE_MS, MAX_SLICE_MS)),
        "interval_ms": float(np.clip(interval_ms, MIN_INTERVAL_MS, MAX_INTERVAL_MS)),
        "repeats": int(repeats),
        "reference_rms": float(ref_rms),
    }


def _select_words(words: list[dict[str, Any]], density: float) -> list[dict[str, Any]]:
    """
    Density controls how many words receive the effect.

    1.0 = every detected word.
    0.5 = approximately every other detected word.
    """
    if not words:
        return []

    density = float(np.clip(density, 0.2, 1.0))

    if density >= 0.999:
        return words

    count = max(1, round(len(words) * density))
    indices = np.linspace(0, len(words) - 1, count).round().astype(int)
    return [words[int(i)] for i in np.unique(indices)]


def _render_word_stutters(
    dry: np.ndarray,
    sr: int,
    words: list[dict[str, Any]],
    style: dict[str, float | int],
    wet_db: float,
    strength: float,
    crossfade_ms: float,
) -> np.ndarray:
    """
    Add stutters after each detected word end.

    The dry vocal remains untouched. One global wet-layer normalization is used
    so individual words do not jump up and down in volume.
    """
    slice_ms = float(style["slice_ms"])
    interval_ms = float(style["interval_ms"])
    repeats = int(style["repeats"])

    wet_db = float(np.clip(wet_db, -18.0, 0.0))
    strength = float(np.clip(strength, 0.25, 1.5))
    crossfade_ms = float(np.clip(crossfade_ms, 3.0, 20.0))

    slice_samples = max(64, int(sr * slice_ms / 1000.0))
    interval_samples = max(slice_samples, int(sr * interval_ms / 1000.0))
    first_offset_samples = max(1, int(sr * FIRST_REPEAT_OFFSET_MS / 1000.0))
    fade_samples = max(1, int(sr * crossfade_ms / 1000.0))

    # Extra room for repeat tails after the final word.
    extra_tail = int(sr * 1.0)
    output = np.pad(dry.astype(np.float32), (0, extra_tail))
    wet = np.zeros_like(output, dtype=np.float32)

    for word in words:
        word_start = int(float(word["start"]) * sr)
        word_end = int(float(word["end"]) * sr)

        # Always anchor to the word end. Take the final part of that word.
        source_end = min(len(dry), max(0, word_end))
        source_start = max(word_start, source_end - slice_samples)

        snippet = dry[source_start:source_end].copy()

        if len(snippet) < max(32, int(sr * 0.025)):
            continue

        snippet -= float(np.mean(snippet))

        fade_len = min(fade_samples, max(1, len(snippet) // 6))
        fade_in, fade_out = _equal_power_fades(fade_len)
        snippet[:fade_len] *= fade_in
        snippet[-fade_len:] *= fade_out

        first_repeat = word_end + first_offset_samples

        for repeat_index in range(repeats):
            start = first_repeat + repeat_index * interval_samples
            end = min(len(wet), start + len(snippet))

            if start >= len(wet) or end <= start:
                break

            decay_db = REPEAT_DECAY_DB[
                min(repeat_index, len(REPEAT_DECAY_DB) - 1)
            ]
            repeat_gain = _db_to_amp(decay_db)

            wet[start:end] += snippet[:end - start] * repeat_gain * strength

    # One global gain for the whole stutter layer prevents word-to-word pumping.
    dry_rms = _rms(dry)
    wet_rms = _rms(wet)

    if wet_rms > 1e-9 and dry_rms > 1e-9:
        target_wet_rms = dry_rms * 0.72
        global_gain = float(np.clip(target_wet_rms / wet_rms, 0.25, 1.5))
        wet *= global_gain

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
    """
    Drop-in replacement expected by app.py.

    The dry vocal is transcribed for true word timestamps, then a retrigger is
    placed after each selected word end. The reference is used only for style.
    """
    del prompt

    with tempfile.TemporaryDirectory(prefix="vst_word_v2_") as temp_dir:
        temp = Path(temp_dir)

        reference_wav = temp / "reference.wav"
        dry_wav = temp / "dry.wav"

        logger.info("Converting reference audio")
        _convert_to_wav(reference_path, reference_wav)

        logger.info("Converting dry vocal")
        _convert_to_wav(dry_path, dry_wav)

        reference, reference_sr = _load_mono(reference_wav)
        dry, dry_sr = _load_mono(dry_wav)

        if reference_sr != dry_sr:
            raise ValueError("Internal sample-rate mismatch.")

        words = _transcribe_word_ends(dry_wav)

        if not words:
            raise ValueError(
                "No words were detected in the dry vocal. "
                "Use a clearer isolated vocal or reduce background music."
            )

        selected_words = _select_words(words, density)
        style = _estimate_style_from_reference(reference, reference_sr)

        logger.info(
            "Rendering %d word-aligned stutter events",
            len(selected_words),
        )

        processed = _render_word_stutters(
            dry=dry,
            sr=dry_sr,
            words=selected_words,
            style=style,
            wet_db=wet_db,
            strength=strength,
            crossfade_ms=crossfade_ms,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, processed, dry_sr, subtype="PCM_24")

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("The output WAV was not created correctly.")

    return {
        "event_count": len(selected_words),
        "word_count": len(words),
        "reference_duration": round(len(reference) / reference_sr, 3),
        "dry_duration": round(len(dry) / dry_sr, 3),
        "model": MODEL_SIZE,
        "first_repeat_offset_ms": FIRST_REPEAT_OFFSET_MS,
        "interval_ms": style["interval_ms"],
        "slice_ms": style["slice_ms"],
        "repeats": style["repeats"],
    }
