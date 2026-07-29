
from pathlib import Path
import subprocess, tempfile, uuid, json
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
JOBS = BASE / "jobs"
JOBS.mkdir(exist_ok=True)
SR = 44100

app = FastAPI()

def convert(src: Path, dst: Path):
    subprocess.run([
        "ffmpeg","-y","-hide_banner","-loglevel","error",
        "-i",str(src),"-vn","-ac","1","-ar",str(SR),
        "-c:a","pcm_s16le",str(dst)
    ], check=True)

def load_any(path: Path):
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "x.wav"
        convert(path, wav)
        y, sr = sf.read(wav)
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)
        if sr != SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=SR)
        peak = np.max(np.abs(y)) if len(y) else 0
        return y / peak * .98 if peak > 1e-9 else y

def onsets(y):
    hop = 256
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=SR, hop_length=hop,
        backtrack=True, units="frames"
    )
    return librosa.frames_to_time(frames, sr=SR, hop_length=hop)

def detect_events(ref):
    times = onsets(ref)
    events = []
    slices = [32,40,48,56,64,80]
    gaps = [44,52,60,68,76,88,100,116,132]
    for t in times:
        c = int(t*SR)
        best = None
        for sm in slices:
            n = int(sm*SR/1000)
            a = c-n
            if a < 0: continue
            base = ref[a:c]
            if np.sqrt(np.mean(base**2)+1e-12) < .004: continue
            for gm in gaps:
                g = int(gm*SR/1000)
                sims = []
                for r in range(1,5):
                    b = a+r*g
                    if b+n > len(ref): break
                    seg = ref[b:b+n]
                    num = abs(np.dot(base,seg))
                    den = np.linalg.norm(base)*np.linalg.norm(seg)+1e-9
                    sims.append(float(num/den))
                good = sum(s>.60 for s in sims)
                conf = float(np.mean(sims)) if sims else 0
                if good >= 2 and (best is None or conf > best["confidence"]):
                    best = dict(time=float(t), slice_ms=sm, interval_ms=gm,
                                repeats=min(4,good), confidence=conf)
        if best:
            events.append(best)
    if not events:
        events = [dict(time=float(t),slice_ms=48,interval_ms=68,repeats=3,confidence=.25)
                  for t in times[::3][:80]]
    return events[:120]

def render(dry, events, wet_db, density, strength, fade_ms):
    dry_times = onsets(dry)
    dur = len(dry)/SR
    out = dry.copy()
    stride = max(1, round(1/max(.05,density)))
    wet = 10**(wet_db/20)*strength
    fade = int(fade_ms*SR/1000)

    for i,e in enumerate(events):
        if i % stride: continue
        t = (e["time"]/max(events[-1]["time"],.01))*dur
        if len(dry_times):
            near = dry_times[np.abs(dry_times-t) < .22]
            if len(near):
                t = float(near[np.argmin(np.abs(near-t))] + .055)
        end = int(np.clip(t,0,dur-.02)*SR)
        n = max(64,int(e["slice_ms"]*SR/1000))
        start = max(0,end-n)
        frag = dry[start:end].copy()
        if len(frag)<32: continue
        frag -= frag.mean()
        gap = int(e["interval_ms"]*SR/1000)
        for r in range(e["repeats"]):
            dst = end+r*gap
            gain = wet*(10**((-2*r)/20))
            for j,v in enumerate(frag):
                k = dst+j
                if k>=len(out): break
                g = 1.0
                if j<fade: g*=j/max(1,fade)
                if j>len(frag)-fade: g*=(len(frag)-j)/max(1,fade)
                out[k] += v*gain*max(0,g)
    p = np.max(np.abs(out)) if len(out) else 0
    if p>.98: out *= .98/p
    return out

async def save_upload(upload, path):
    with path.open("wb") as f:
        while chunk := await upload.read(1024*1024):
            f.write(chunk)

@app.post("/api/process")
async def process(
    reference: UploadFile=File(...),
    dry_vocal: UploadFile=File(...),
    prompt: str=Form("Clean micro stutters, subtle, no pitch shift, wet -8 dB."),
    wet_db: float=Form(-8),
    density: float=Form(1),
    strength: float=Form(1),
    crossfade_ms: float=Form(8),
):
    job = uuid.uuid4().hex
    d = JOBS/job
    d.mkdir()
    refp=d/("ref"+Path(reference.filename or ".m4a").suffix)
    dryp=d/("dry"+Path(dry_vocal.filename or ".m4a").suffix)
    outp=d/"Vocal_Stutter_Transfer.wav"
    try:
        await save_upload(reference,refp)
        await save_upload(dry_vocal,dryp)
        ref=load_any(refp); dry=load_any(dryp)
        p=prompt.lower()
        if "subtle" in p: wet_db=min(wet_db,-10); strength=min(strength,.75)
        if "strong" in p: wet_db=max(wet_db,-6); strength=max(strength,1.2)
        if "sparse" in p or "fewer" in p: density=.45
        events=detect_events(ref)
        out=render(dry,events,wet_db,density,strength,crossfade_ms)
        sf.write(outp,out,SR,subtype="PCM_24")
        (d/"analysis.json").write_text(json.dumps(events,indent=2))
        return {"download_url":f"/api/download/{job}","event_count":len(events)}
    except Exception as e:
        raise HTTPException(500,str(e))

@app.get("/api/download/{job}")
def download(job: str):
    p=JOBS/job/"Vocal_Stutter_Transfer.wav"
    if not p.exists(): raise HTTPException(404)
    return FileResponse(p,media_type="audio/wav",filename=p.name)

# Ensure the static directory exists before mounting it.
STATIC = BASE / "static"
STATIC.mkdir(parents=True, exist_ok=True)

app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")