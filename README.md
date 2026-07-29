# Vocal Stutter Transfer — Full Replacement

A complete FastAPI + FFmpeg project designed for Render.

## Upload these files to the root of a new GitHub repository

- `app.py`
- `audio_processing.py`
- `Dockerfile`
- `requirements.txt`
- `.dockerignore`
- `static/` with its three files inside

Do not upload the enclosing project folder as an extra nested level.

## Render

1. New Web Service
2. Connect the new repository
3. Runtime: Docker
4. Deploy

Health check: `/api/health`

The engine processes at 48 kHz and exports 24-bit WAV. Default wet mix is -3 dB and default strength is 115%.
