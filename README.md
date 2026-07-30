# Vocal Stutter Transfer — Async Render Fix

This version avoids Render/iPhone 502 timeouts by returning a job ID immediately and processing audio in a background worker. The browser polls `/api/status/{job_id}` until the WAV is ready.

## Render
- Runtime: Docker
- Root Directory: leave blank
- Dockerfile Path: `Dockerfile`
- Use one instance and one worker (the included command already does this)

Upload the files from this folder directly to the GitHub repository root.
