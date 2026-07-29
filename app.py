from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

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
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/tmp/vocal_stutter_outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Vocal Stutter Transfer", version="1.0.0")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/process")
async def process_audio(
    reference: UploadFile = File(...),
    dry_vocal: UploadFile = File(...),
    prompt: str = Form("Clean micro stutters, subtle, no pitch shift, wet -8 dB."),
    wet_db: float = Form(-8.0),
    density: float = Form(1.0),
    strength: float = Form(1.0),
    crossfade_ms: float = Form(8.0),
) -> dict:
    job_id = uuid.uuid4().hex
    job_dir = Path(tempfile.mkdtemp(prefix=f"vst_{job_id}_"))
    logger.info("Job %s started", job_id)

    try:
        reference_path = job_dir / (reference.filename or "reference_audio")
        dry_path = job_dir / (dry_vocal.filename or "dry_vocal")

        with reference_path.open("wb") as f:
            shutil.copyfileobj(reference.file, f)

        with dry_path.open("wb") as f:
            shutil.copyfileobj(dry_vocal.file, f)

        output_path = OUTPUT_DIR / f"{job_id}.wav"

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

        logger.info(
            "Job %s completed with %s detected events",
            job_id,
            result["event_count"],
        )

        return {
            "job_id": job_id,
            "event_count": result["event_count"],
            "reference_duration": result["reference_duration"],
            "dry_duration": result["dry_duration"],
            "download_url": f"/api/download/{job_id}",
        }

    except ValueError as exc:
        logger.exception("Job %s rejected", job_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


@app.get("/api/download/{job_id}")
def download(job_id: str):
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Invalid job ID.")

    path = OUTPUT_DIR / f"{job_id}.wav"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Processed file not found.")

    return FileResponse(
        path,
        media_type="audio/wav",
        filename="processed_vocal_stutter.wav",
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
