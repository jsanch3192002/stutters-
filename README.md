# Vocal Stutter Transfer

A Render-ready FastAPI app that analyzes repeated micro-stutter timing in a processed reference vocal and transfers similar retrigger events onto a dry vocal.

## Project structure

```text
vocal_stutter_transfer/
├── app.py
├── audio_processing.py
├── Dockerfile
├── requirements.txt
└── static/
    ├── index.html
    ├── app.js
    └── style.css
```

## Deploy to Render

1. Create a new GitHub repository.
2. Upload the **contents** of this project folder, not the ZIP itself.
3. In Render, create a new **Web Service** from that repository.
4. Choose **Docker** as the runtime.
5. Leave the Docker command blank.
6. Deploy.
7. After it becomes Live, open the Render URL.

## Important limits

- Maximum audio length: 5 minutes per uploaded file.
- Short test files (10–30 seconds) are recommended for the first deployment test.
- Render free instances can sleep and may take extra time on the first request.
- The algorithm preserves pitch by copying audio slices without time-stretching or repitching.
