const form = document.getElementById("form");
const reference = document.getElementById("reference");
const dryVocal = document.getElementById("dryVocal");
const processBtn = document.getElementById("processBtn");
const progress = document.getElementById("progress");
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
  input.addEventListener("input", () => output.textContent = formatter(Number(input.value)));
}

function showStatus(message, type = "") {
  statusBox.hidden = false;
  statusBox.className = `status ${type}`.trim();
  statusBox.textContent = message;
}

form.addEventListener("submit", async event => {
  event.preventDefault();

  if (!reference.files[0] || !dryVocal.files[0]) {
    showStatus("Select both audio files first.", "error");
    return;
  }

  processBtn.disabled = true;
  progress.hidden = false;
  downloadBtn.hidden = true;
  showStatus("Uploading and processing. This can take a few minutes on Render.");

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
    let data = {};
    try {
      data = await response.json();
    } catch (_) {}

    if (!response.ok) {
      throw new Error(data.detail || `Server error (${response.status})`);
    }

    downloadBtn.href = data.download_url;
    downloadBtn.hidden = false;
    showStatus(
      `Finished. Detected ${data.event_count} reference stutter event${data.event_count === 1 ? "" : "s"}.`,
      "success"
    );
  } catch (error) {
    showStatus(error.message || "Processing failed.", "error");
  } finally {
    processBtn.disabled = false;
    progress.hidden = true;
  }
});
