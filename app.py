from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from audio_processing import process_stutter_transfer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("vocal-stutter-transfer")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
WORK_DIR = Path(os.environ.get("WORK_DIR", "/tmp/vocal_stutter_jobs"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(150 * 1024 * 1024)))

STATIC_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vocal Stutter Transfer", version="3.0.0")
executor = ThreadPoolExecutor(max_workers=max(1, int(os.environ.get("AUDIO_WORKERS", "1"))))
jobs: dict[str, dict[str, Any]] = {}
jobs_lock = threading.Lock()


def set_job(job_id: str, **changes: Any) -> None:
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(changes)


def get_job(job_id: str) -> dict[str, Any] | None:
    with jobs_lock:
        value = jobs.get(job_id)
        return dict(value) if value else None


def safe_name(filename: str | None, fallback: str) -> str:
    name = Path(filename or fallback).name
    return name or fallback


def save_upload(upload: UploadFile, destination: Path) -> int:
    written = 0
    with destination.open("wb") as target:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise ValueError("An uploaded file is too large. Maximum size is 150 MB per file.")
            target.write(chunk)
    if written == 0:
        raise ValueError("One of the uploaded files is empty.")
    return written


def run_job(
    job_id: str,
    reference_path: Path,
    dry_path: Path,
    output_path: Path,
    wet_db: float,
    density: float,
    strength: float,
    crossfade_ms: float,
    prompt: str,
) -> None:
    try:
        set_job(job_id, status="processing", progress=15, message="Analyzing the reference audio…")
        logger.info("Job %s started", job_id)

        result = process_stutter_transfer(
            reference_path=reference_path,
            dry_path=dry_path,
            output_path=output_path,
            wet_db=wet_db,
            density=density,
            strength=strength,
            crossfade_ms=crossfade_ms,
            prompt=prompt,
        )

        if not output_path.is_file():
            raise RuntimeError("Processing finished but no output file was created.")
        output_size = output_path.stat().st_size
        if output_size < 44:
            raise RuntimeError("The processed WAV file is empty or invalid.")

        set_job(
            job_id,
            status="complete",
            progress=100,
            message="Finished.",
            result=result,
            output_size=output_size,
            download_url=f"/api/download/{job_id}",
        )
        logger.info("Job %s completed: %s bytes", job_id, output_size)
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        set_job(
            job_id,
            status="failed",
            progress=100,
            message=f"Processing failed: {type(exc).__name__}: {exc}",
        )
    finally:
        reference_path.unlink(missing_ok=True)
        dry_path.unlink(missing_ok=True)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/process", status_code=202)
def process_audio(
    reference: UploadFile = File(...),
    dry_vocal: UploadFile = File(...),
    prompt: str = Form("Clean loud retrigger stutters with no pitch shift."),
    wet_db: float = Form(-3.0),
    density: float = Form(1.0),
    strength: float = Form(1.15),
    crossfade_ms: float = Form(8.0),
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    job_dir = WORK_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    reference_path = job_dir / safe_name(reference.filename, "reference_audio")
    dry_path = job_dir / safe_name(dry_vocal.filename, "dry_vocal")
    output_path = job_dir / "processed_vocal_stutter.wav"

    try:
        reference_size = save_upload(reference, reference_path)
        dry_size = save_upload(dry_vocal, dry_path)
    except Exception:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        reference.file.close()
        dry_vocal.file.close()

    set_job(
        job_id,
        status="queued",
        progress=5,
        message="Upload complete. Waiting to process…",
        output_path=str(output_path),
        reference_size=reference_size,
        dry_size=dry_size,
    )

    executor.submit(
        run_job,
        job_id,
        reference_path,
        dry_path,
        output_path,
        float(wet_db),
        float(density),
        float(strength),
        float(crossfade_ms),
        prompt,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/status/{job_id}",
    }


@app.get("/api/status/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found. The server may have restarted.")

    response: dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "message": job.get("message", ""),
    }
    if job.get("status") == "complete":
        response.update(job.get("result") or {})
        response["download_url"] = job.get("download_url")
        response["output_size"] = job.get("output_size")
    return response


@app.get("/api/download/{job_id}")
def download(job_id: str) -> FileResponse:
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job ID.")
    job = get_job(job_id)
    if not job or job.get("status") != "complete":
        raise HTTPException(status_code=404, detail="Processed file is not ready or no longer exists.")

    path = Path(str(job.get("output_path", "")))
    if not path.is_file() or path.stat().st_size < 44:
        raise HTTPException(status_code=404, detail="Processed file not found.")

    return FileResponse(
        path,
        media_type="audio/wav",
        filename="processed_vocal_stutter.wav",
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
