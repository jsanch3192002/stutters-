const form = document.getElementById("form");
const reference = document.getElementById("reference");
const dryVocal = document.getElementById("dryVocal");
const processBtn = document.getElementById("processBtn");
const progressWrap = document.getElementById("progress");
const progressBar = document.getElementById("progressBar");
const statusBox = document.getElementById("status");
const downloadBtn = document.getElementById("downloadBtn");

function fileLabel(file) {
  if (!file) return "No file selected";
  const mb = file.size / (1024 * 1024);
  return `${file.name} • ${mb.toFixed(1)} MB`;
}

reference.addEventListener("change", () => {
  document.getElementById("referenceName").textContent = fileLabel(reference.files[0]);
});

dryVocal.addEventListener("change", () => {
  document.getElementById("dryName").textContent = fileLabel(dryVocal.files[0]);
});

const sliders = [
  ["wetDb", "wetDbValue", v => `${v} dB`],
  ["density", "densityValue", v => `${Math.round(v * 100)}%`],
  ["strength", "strengthValue", v => `${Math.round(v * 100)}%`],
  ["crossfade", "crossfadeValue", v => `${v} ms`]
];

for (const [id, outputId, formatter] of sliders) {
  const input = document.getElementById(id);
  const output = document.getElementById(outputId);
  input.addEventListener("input", () => {
    output.textContent = formatter(Number(input.value));
  });
}

function showStatus(message, type = "") {
  statusBox.hidden = false;
  statusBox.className = `status ${type}`.trim();
  statusBox.textContent = message;
}

function setProgress(value) {
  const safe = Math.max(0, Math.min(100, Number(value) || 0));
  progressWrap.hidden = false;
  if (progressBar) progressBar.style.width = `${safe}%`;
}

async function readJson(response) {
  const text = await response.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`Server returned an invalid response (${response.status}).`);
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function pollJob(statusUrl) {
  const started = Date.now();
  const maxWaitMs = 15 * 60 * 1000;

  while (Date.now() - started < maxWaitMs) {
    const response = await fetch(statusUrl, { cache: "no-store" });
    const data = await readJson(response);

    if (!response.ok) {
      throw new Error(data.detail || `Status check failed (${response.status}).`);
    }

    setProgress(data.progress || 10);
    showStatus(data.message || "Processing…");

    if (data.status === "complete") return data;
    if (data.status === "failed") {
      throw new Error(data.message || "Processing failed.");
    }

    await sleep(2000);
  }

  throw new Error("Processing took longer than 15 minutes.");
}

form.addEventListener("submit", async event => {
  event.preventDefault();

  if (!reference.files[0] || !dryVocal.files[0]) {
    showStatus("Select both audio files first.", "error");
    return;
  }

  processBtn.disabled = true;
  downloadBtn.hidden = true;
  downloadBtn.removeAttribute("href");
  setProgress(2);
  showStatus("Uploading files…");

  const body = new FormData();
  body.append("reference", reference.files[0]);
  body.append("dry_vocal", dryVocal.files[0]);
  body.append("prompt", "Clean micro stutters, loud, clean, no pitch shift.");
  body.append("wet_db", document.getElementById("wetDb").value);
  body.append("density", document.getElementById("density").value);
  body.append("strength", document.getElementById("strength").value);
  body.append("crossfade_ms", document.getElementById("crossfade").value);

  try {
    const response = await fetch("/api/process", { method: "POST", body });
    const queued = await readJson(response);

    if (!response.ok) {
      throw new Error(queued.detail || `Upload failed (${response.status}).`);
    }
    if (!queued.status_url) {
      throw new Error("The server did not return a job status URL.");
    }

    setProgress(8);
    showStatus("Upload complete. Processing in the background…");
    const result = await pollJob(queued.status_url);

    if (!result.download_url) {
      throw new Error("Processing finished, but no download URL was returned.");
    }

    downloadBtn.href = result.download_url;
    downloadBtn.hidden = false;
    setProgress(100);
    showStatus(
      `Finished. Detected ${result.event_count ?? 0} reference stutter event${result.event_count === 1 ? "" : "s"}.`,
      "success"
    );
  } catch (error) {
    console.error(error);
    showStatus(error?.message || "Processing failed.", "error");
    setProgress(0);
  } finally {
    processBtn.disabled = false;
  }
});
