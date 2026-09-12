// Clipping Machine dev console.
// Vanilla JS, no build step, served same-origin by the FastAPI app (see
// app/main.py's StaticFiles mount) -- so no CORS setup needed and no
// framework/toolchain required to run this. Talks directly to the real
// API endpoints documented in the architecture doc.

const API_BASE = window.location.origin;
const TOKEN_KEY = "cm_token";

const STATUS_CLASS = {
  queued: "pill-warn",
  ingesting: "pill-warn",
  ingested: "pill-warn",
  transcribing: "pill-warn",
  transcribed: "pill-warn",
  segmenting: "pill-warn",
  segmented: "pill-warn",
  scoring: "pill-warn",
  scored: "pill-warn",
  rendering: "pill-warn",
  ready_for_review: "pill-ok",
  rendered: "pill-ok",
  approved: "pill-ok",
  uploaded: "pill-ok",
  archived: "pill-unknown",
  pending: "pill-unknown",
  blocked: "pill-unknown",
};
function statusClass(status) {
  if (STATUS_CLASS[status]) return STATUS_CLASS[status];
  if (status && status.startsWith("failed")) return "pill-err";
  return "pill-unknown";
}

// score is 0..10 (app.core.scoring_logic.score_out_of_10) -- a deterministic
// heuristic ranking, not a virality prediction (see project AI/ML
// principles). Color bands are just a quick visual read, not a claim of
// precision at the 0.1 level.
function scoreClass(score) {
  if (score >= 6.5) return "pill-ok";
  if (score >= 3.5) return "pill-warn";
  return "pill-unknown";
}

function scoreTooltip(breakdown) {
  if (!breakdown?.raw) return "Score breakdown unavailable";
  const lines = Object.entries(breakdown.raw).map(
    ([feature, raw]) => `${feature}: raw=${Number(raw).toFixed(2)}, weight=${Number(breakdown.weights?.[feature] ?? 0).toFixed(2)}`
  );
  return `Heuristic score, not a virality prediction.\n${lines.join("\n")}`;
}

let selectedJobId = null;
let creatorAccountsCache = [];
let autoRefreshTimer = null;
let lastRenderedClipsSignature = null;

// ---- session ----
//
// Auth is a real login now (app/api/routers/auth.py). The browser holds an
// httpOnly session cookie, which JavaScript deliberately cannot read -- so
// there is nothing to store here and nothing an XSS bug could steal. Every
// request just carries the cookie automatically.
//
// getToken() survives for one reason: a bearer token pasted into
// localStorage by hand still works, which keeps scripts, curl and the
// README examples usable against the same API.

let currentUser = null;

function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return ""; // storage disabled -- the cookie is doing the work anyway
  }
}

/**
 * Gate the console on a real session.
 *
 * Returns true when signed in. On 401 it sends the browser to the login
 * page and returns false, so the caller must stop -- otherwise every
 * subsequent request fires and 401s during the redirect.
 *
 * With dev auto-login enabled server-side this simply succeeds and no login
 * page is ever seen, which is the point: local dev keeps its zero-friction
 * flow while a deployed instance is properly gated.
 */
async function requireAuth() {
  let resp;
  try {
    resp = await fetch(`${API_BASE}/api/v1/auth/me`, {
      headers: getToken() ? { Authorization: `Bearer ${getToken()}` } : {},
    });
  } catch {
    // The API is unreachable. Redirecting to the login page would just show
    // a form that also cannot reach it, so stay put and say what is wrong.
    // Setting the pill directly rather than calling a helper: this runs
    // before anything else and must not depend on other startup code.
    const pill = $("#conn-status");
    if (pill) {
      pill.textContent = "API unreachable — is uvicorn running?";
      pill.className = "pill pill-err";
    }
    return false;
  }

  if (resp.status === 401) {
    window.location.href = "/login.html";
    return false;
  }
  if (!resp.ok) return false;

  currentUser = await resp.json();
  const devBypass = currentUser.auth_source === "dev_auto_login";

  const label = $("#signed-in-as");
  if (label) {
    label.textContent = devBypass ? `${currentUser.email} (dev auto-login)` : currentUser.email;
    label.title = devBypass
      ? "DEV_AUTO_LOGIN_EMAIL is set in .env, so the API accepts every request as this user and " +
        "ignores cookies. Comment it out and restart the API to use real logins."
      : "";
  }
  // Sign out cannot work under dev auto-login -- it clears a cookie the
  // server is not consulting. Hiding the button is more honest than offering
  // one that appears to do nothing.
  const btn = $("#logout-btn");
  if (btn) btn.hidden = devBypass;
  return true;
}

async function logout() {
  try {
    await fetch(`${API_BASE}/api/v1/auth/logout`, { method: "POST" });
  } catch {
    /* even if the call fails, clear local state and send them to the form */
  }
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* nothing to clear */
  }
  window.location.href = "/login.html";
}

function initSessionUi() {
  $("#logout-btn")?.addEventListener("click", logout);
}

// ---- fetch helper ----

async function apiFetch(path, options = {}) {
  const token = getToken();
  const headers = options.headers || {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const resp = await fetch(`${API_BASE}${path}`, { ...options, headers });

  let body = null;
  const text = await resp.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { error: { code: "non_json_response", message: text.slice(0, 300) } };
    }
  }

  if (!resp.ok) {
    const message = body?.error?.message || body?.detail || `HTTP ${resp.status}`;
    const code = body?.error?.code || "http_error";
    showToast(`${code}: ${message}`);
    throw new Error(`${code}: ${message}`);
  }
  return body;
}

function showToast(message) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 6000);
}

function $(sel) {
  return document.querySelector(sel);
}

function escapeHtml(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ---- connection status ----

async function checkHealth() {
  const pill = $("#conn-status");
  try {
    const resp = await fetch(`${API_BASE}/health`);
    if (resp.ok) {
      pill.textContent = "API reachable";
      pill.className = "pill pill-ok";
    } else {
      pill.textContent = `API returned ${resp.status}`;
      pill.className = "pill pill-err";
    }
  } catch {
    pill.textContent = "API unreachable";
    pill.className = "pill pill-err";
  }
}

// ---- jobs ----

async function loadJobs() {
  let jobs;
  try {
    jobs = await apiFetch("/api/v1/stream-jobs?limit=50");
  } catch {
    return;
  }
  $("#jobs-count").textContent = `(${jobs.length})`;
  const body = $("#jobs-body");
  body.innerHTML = "";
  for (const job of jobs) {
    const tr = document.createElement("tr");
    tr.className = "selectable";
    if (job.id === selectedJobId) tr.style.background = "rgba(91,141,239,0.08)";
    tr.innerHTML = `
      <td class="mono">${new Date(job.created_at).toLocaleTimeString()}</td>
      <td><span class="pill ${statusClass(job.status)}">${job.status}</span></td>
      <td>${job.duration_seconds != null ? job.duration_seconds.toFixed(1) + "s" : "—"}</td>
      <td>${job.retry_count}</td>
      <td class="muted">${escapeHtml(job.last_error || "")}</td>
      <td>
        <button class="secondary view-btn" data-job-id="${job.id}">View</button>
        ${isJobRunning(job.status)
          ? `<button class="secondary cancel-job-btn" data-job-id="${job.id}" title="Stop this job at its next stage boundary">Cancel</button>`
          : ""}
        <button class="secondary danger delete-job-btn" data-job-id="${job.id}" title="Delete this upload and its clips">🗑</button>
      </td>
    `;
    tr.querySelector(".view-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      selectJob(job.id);
    });
    tr.querySelector(".cancel-job-btn")?.addEventListener("click", (e) => {
      e.stopPropagation();
      cancelJob(job.id);
    });
    tr.querySelector(".delete-job-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      deleteJob(job.id);
    });
    tr.addEventListener("click", () => selectJob(job.id));
    body.appendChild(tr);
  }
}

async function deleteJob(jobId) {
  if (!confirm("Delete this upload and every clip generated from it? This can't be undone.")) return;
  try {
    await apiFetch(`/api/v1/stream-jobs/${jobId}`, { method: "DELETE" });
  } catch {
    return; // toast already shown
  }
  if (selectedJobId === jobId) {
    selectedJobId = null;
    lastRenderedClipsSignature = null;
    $("#job-detail-panel").hidden = true;
  }
  await loadJobs();
}

async function selectJob(jobId) {
  selectedJobId = jobId;
  lastRenderedClipsSignature = null; // force a fresh render for the newly selected job
  $("#job-detail-panel").hidden = false;
  await loadJobDetail();
  await loadJobs(); // re-render to highlight selection
}

// ---- layout region marker (facecam + gameplay) ----
//
// Face detection had to answer "is there a facecam, and where" -- and a
// centered crop had to guess which third of a 16:9 frame mattered. Both
// guesses were the source of the bad framing this replaces. A creator
// drawing boxes on a frame of their own VOD is the one input that cannot be
// wrong, so a marked region overrides detection, classify_reaction_layout
// and crop_bias for that panel (see app/workers/rendering.py).
//
// Coordinates are NORMALIZED (0..1 of the frame) against the image's own
// displayed size, so the browser never needs to know the source resolution
// and a mark stays correct at any resolution.
const REGION_COLORS = { facecam: "#4ade80", gameplay: "#60a5fa" };
let markerMode = "facecam";
let markedRegions = { facecam: null, gameplay: null }; // normalized, pending save

function facecamSetStatus(message, isError = false) {
  const el = $("#facecam-status");
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? "#f88" : "";
}

function renderMarkedBoxes() {
  const canvas = $("#facecam-stage").querySelector(".facecam-canvas");
  if (!canvas) return;
  canvas.querySelectorAll(".facecam-box.saved").forEach((el) => el.remove());
  const r = canvas.getBoundingClientRect();
  for (const [name, rect] of Object.entries(markedRegions)) {
    if (!rect) continue;
    const el = document.createElement("div");
    el.className = "facecam-box saved";
    el.style.borderColor = REGION_COLORS[name];
    el.style.background = `${REGION_COLORS[name]}22`;
    el.style.left = `${rect.x * r.width}px`;
    el.style.top = `${rect.y * r.height}px`;
    el.style.width = `${rect.w * r.width}px`;
    el.style.height = `${rect.h * r.height}px`;
    const tag = document.createElement("span");
    tag.className = "facecam-box-tag";
    tag.style.background = REGION_COLORS[name];
    tag.textContent = name;
    el.appendChild(tag);
    canvas.appendChild(el);
  }
}

function describeMarks() {
  const bits = [];
  for (const [name, rect] of Object.entries(markedRegions)) {
    if (rect) bits.push(`${name} ${(rect.w * 100).toFixed(0)}%x${(rect.h * 100).toFixed(0)}%`);
  }
  return bits.length ? bits.join("  |  ") : "nothing marked yet";
}

async function loadFacecamFrame() {
  if (!selectedJobId) return;
  const stage = $("#facecam-stage");
  const at = Number($("#facecam-at").value || 0);
  stage.innerHTML = `<div class="muted">Loading frame…</div>`;

  let url;
  try {
    url = await fetchMediaBlobUrl(`/api/v1/stream-jobs/${selectedJobId}/frame?at_seconds=${at}`);
  } catch (err) {
    stage.innerHTML = "";
    facecamSetStatus(
      `Could not load a frame at ${at}s (${err.message}). If the upload never finished, or that ` +
      `timestamp is past the end of the video, try 0.`,
      true,
    );
    return;
  }

  stage.innerHTML = `
    <div class="facecam-canvas">
      <img id="facecam-img" src="${url}" alt="frame from this VOD" draggable="false" />
      <div id="facecam-box" class="facecam-box" hidden></div>
    </div>`;

  const canvas = stage.querySelector(".facecam-canvas");
  const box = $("#facecam-box");
  let dragging = false;
  let originX = 0;
  let originY = 0;

  // Read the rect per drag rather than caching it: the image reflows on
  // window resize and when the panel is re-rendered.
  const relative = (event) => {
    const r = canvas.getBoundingClientRect();
    return {
      x: Math.min(Math.max(event.clientX - r.left, 0), r.width),
      y: Math.min(Math.max(event.clientY - r.top, 0), r.height),
      w: r.width,
      h: r.height,
    };
  };

  const paint = (x, y, w, h) => {
    box.hidden = false;
    box.style.borderColor = REGION_COLORS[markerMode];
    box.style.background = `${REGION_COLORS[markerMode]}22`;
    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
    box.style.width = `${w}px`;
    box.style.height = `${h}px`;
  };

  canvas.addEventListener("pointerdown", (e) => {
    if (e.target.classList.contains("facecam-box-tag")) return;
    const p = relative(e);
    dragging = true;
    originX = p.x;
    originY = p.y;
    canvas.setPointerCapture(e.pointerId);
    paint(p.x, p.y, 0, 0);
    e.preventDefault();
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const p = relative(e);
    paint(Math.min(originX, p.x), Math.min(originY, p.y), Math.abs(p.x - originX), Math.abs(p.y - originY));
  });

  const finishDrag = (e) => {
    if (!dragging) return;
    dragging = false;
    const p = relative(e);
    const x = Math.min(originX, p.x);
    const y = Math.min(originY, p.y);
    const w = Math.abs(p.x - originX);
    const h = Math.abs(p.y - originY);
    box.hidden = true;
    // A stray click is a 0-size box, not an instruction.
    if (w < 8 || h < 8) {
      facecamSetStatus(`That box was too small to count. Drag a box around the ${markerMode}.`);
      return;
    }
    markedRegions[markerMode] = { x: x / p.w, y: y / p.h, w: w / p.w, h: h / p.h };
    $("#facecam-save").disabled = false;
    renderMarkedBoxes();
    facecamSetStatus(`${describeMarks()} — press "Save boxes" to apply.`);
  };

  canvas.addEventListener("pointerup", finishDrag);
  canvas.addEventListener("pointercancel", finishDrag);

  renderMarkedBoxes();
  facecamSetStatus(`Drag a box around the ${markerMode}. ${describeMarks()}`);
}

async function saveLayoutRegions(regions) {
  if (!selectedJobId) return;
  try {
    await apiFetch(`/api/v1/stream-jobs/${selectedJobId}/layout-regions`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(regions),
    });
    facecamSetStatus(
      regions.facecam || regions.gameplay
        ? "Saved. The next render of this job's clips uses these boxes (already-rendered clips keep their framing)."
        : "Cleared — this job is back to automatic detection.",
    );
    await loadJobDetail();
  } catch {
    /* toast already shown */
  }
}

function setMarkerMode(mode) {
  markerMode = mode;
  ["facecam", "gameplay"].forEach((m) => {
    const btn = $(`#mode-${m}`);
    if (btn) btn.classList.toggle("active", m === mode);
  });
  facecamSetStatus(`Drag a box around the ${mode}. ${describeMarks()}`);
}

function initFacecamMarker() {
  // Defensive: if index.html is a stale cached copy without this panel,
  // these lookups return null and an unguarded addEventListener would throw
  // during startup -- taking the REST of the console's init down with it.
  const loadBtn = $("#facecam-load");
  if (!loadBtn) {
    console.warn("[clipping-machine] framing panel markup missing -- hard-refresh the page (Ctrl+Shift+R)");
    return;
  }
  loadBtn.addEventListener("click", loadFacecamFrame);
  $("#mode-facecam").addEventListener("click", () => setMarkerMode("facecam"));
  $("#mode-gameplay").addEventListener("click", () => setMarkerMode("gameplay"));
  $("#facecam-save").addEventListener("click", () => saveLayoutRegions(markedRegions));
  $("#facecam-clear").addEventListener("click", () => {
    markedRegions = { facecam: null, gameplay: null };
    $("#facecam-save").disabled = true;
    renderMarkedBoxes();
    saveLayoutRegions({ facecam: null, gameplay: null });
  });
  window.addEventListener("resize", renderMarkedBoxes);
}

// Terminal statuses -- the work is over, so there is nothing to cancel and
// no Cancel button worth showing. Mirrors TERMINAL_JOB_STATUSES in
// app/api/routers/stream_jobs.py; the API refuses the call either way, this
// just avoids offering a button that can only fail.
const TERMINAL_JOB_STATUSES = new Set([
  "ready_for_review", "archived", "cancelled",
  "failed_ingest", "failed_transcription", "failed_segmentation",
  "failed_scoring", "failed_rendering",
]);

function isJobRunning(status) {
  return !TERMINAL_JOB_STATUSES.has(status);
}

async function cancelJob(jobId) {
  if (!confirm(
    "Cancel this job?\n\nIt stops at the next stage boundary, so whatever is running " +
    "right now (a transcription pass, a render) finishes first — that can take a few " +
    "minutes on a long VOD. Clips already rendered are kept."
  )) return;
  try {
    await apiFetch(`/api/v1/stream-jobs/${jobId}/cancel`, { method: "POST" });
    showToast("Cancelling — the current stage will finish, then the job stops.");
    await loadJobs();
    if (selectedJobId === jobId) await loadJobDetail();
  } catch {
    /* toast already shown */
  }
}

async function loadJobDetail() {
  if (!selectedJobId) return;
  let job;
  try {
    job = await apiFetch(`/api/v1/stream-jobs/${selectedJobId}`);
  } catch {
    return;
  }
  $("#detail-job-id").textContent = job.id;
  $("#detail-job-json").textContent = JSON.stringify(job, null, 2);

  // Seed the picker from whatever this job already has stored, so reopening
  // a job shows its marks instead of a blank slate.
  markedRegions = { facecam: job.facecam_rect || null, gameplay: job.gameplay_rect || null };
  const saveBtn = $("#facecam-save");
  if (saveBtn) saveBtn.disabled = !(markedRegions.facecam || markedRegions.gameplay);
  renderMarkedBoxes();

  let clips = [];
  try {
    clips = await apiFetch(`/api/v1/stream-jobs/${selectedJobId}/clips`);
  } catch {
    /* toast already shown */
  }

  const emptyNote = $("#clips-empty-note");
  const list = $("#clips-list");

  if (clips.length === 0) {
    lastRenderedClipsSignature = null;
    list.innerHTML = "";
    emptyNote.textContent =
      "No clips yet -- clips appear once this job reaches 'scored' (scoring selects the top candidates " +
      "and rendering starts on each one automatically).";
    return;
  }
  emptyNote.textContent = "";

  // Skip the rebuild entirely if nothing changed since last render. Without
  // this, the 3s auto-refresh wipes out any in-progress interaction inside
  // a clip card -- an open <select>, a just-clicked button's result text --
  // every few seconds, which makes the panel unusable while polling is on.
  const signature = JSON.stringify(
    clips.map((c) => [
      c.id, c.status, c.caption_text, c.flag_reasons, c.score, c.caption_hashtags, c.caption_source, c.caption_title,
    ])
  );
  if (signature === lastRenderedClipsSignature) return;
  lastRenderedClipsSignature = signature;

  list.innerHTML = "";
  for (const clip of clips) {
    list.appendChild(renderClipCard(clip));
  }
}

function renderClipCard(clip) {
  const card = document.createElement("div");
  card.className = "clip-card";

  const flagText = clip.flag_reasons?.length ? ` · flags: ${clip.flag_reasons.join(", ")}` : "";
  const durationText = clip.duration_seconds != null ? ` · ${clip.duration_seconds.toFixed(1)}s` : "";
  const hasVideo = !!clip.object_key;
  const scoreBadge = clip.score != null
    ? `<span class="pill ${scoreClass(clip.score)}" title="${escapeHtml(scoreTooltip(clip.score_breakdown))}">${clip.score.toFixed(1)}/10</span>`
    : "";

  // Title/hashtags/explanation come from app.core.caption_generation, run
  // synchronously as part of rendering -- present as soon as status hits
  // 'rendered' for any new clip. Only a clip rendered before this field
  // existed would show null here.
  const hasTitle = !!clip.caption_title;
  const titleHtml = hasTitle
    ? `<div class="clip-title-row">
         <div class="clip-title" title="Burned into the top of the video (if title overlay was enabled at render time)">${escapeHtml(clip.caption_title)}</div>
         <button class="link-btn edit-title-btn" title="Edit the stored title text -- does NOT re-render the video">✏️ edit</button>
       </div>
       <div class="title-edit-form" style="display:none">
         <input type="text" class="title-edit-input" placeholder="Bold on-screen hook title">
         <button class="secondary save-title-btn">Save</button>
         <button class="secondary cancel-title-btn">Cancel</button>
       </div>`
    : "";

  // Hashtags/explanation come from the same call -- null only for a clip
  // rendered before this feature existed.
  const hasHashtags = !!clip.caption_hashtags?.length;
  const hashtagsHtml = hasHashtags
    ? `<div class="hashtags-row">
         <div class="hashtags">${clip.caption_hashtags.map((t) => `<span class="hashtag">${escapeHtml(t)}</span>`).join(" ")}</div>
         <button class="link-btn edit-hashtags-btn" title="Edit hashtags before upload">✏️ edit</button>
       </div>
       <div class="hashtag-edit-form" style="display:none">
         <input type="text" class="hashtag-edit-input" placeholder="#tag1 #tag2 #tag3">
         <button class="secondary save-hashtags-btn">Save</button>
         <button class="secondary cancel-hashtags-btn">Cancel</button>
       </div>`
    : clip.status === "rendered"
      ? `<div class="muted small">(no hashtags -- this clip predates hashtag generation)</div>`
      : "";
  // Hover this badge to confirm which model actually produced this specific
  // clip's title/hashtags -- e.g. after switching CAPTION_LLM_PROVIDER, this
  // is the way to check it actually took effect for a given clip rather
  // than just trusting the .env setting (see app.core.caption_generation /
  // RenderedClip.caption_model+caption_reason for where this comes from).
  const sourceLabel = { llm: "✨ AI", manual_edit: "✏️ edited" }[clip.caption_source] || "heuristic";
  let sourceTitle;
  if (clip.caption_source === "llm") {
    sourceTitle = `AI-generated from this clip's transcript (model: ${clip.caption_model || "unknown"})`;
  } else if (clip.caption_source === "manual_edit") {
    sourceTitle = "Manually edited by a reviewer";
  } else if (clip.caption_reason) {
    sourceTitle = `Fallback (not model-generated) -- reason: ${clip.caption_reason}`;
  } else {
    sourceTitle = "Fallback: LLM captions are off, unavailable, or the call failed -- see README's Clip hashtags & captions";
  }
  const sourceBadge = clip.caption_source
    ? `<span class="pill pill-unknown" title="${escapeHtml(sourceTitle)}">${sourceLabel}</span>`
    : "";
  // Only shown for an LLM-suggested candidate window (see
  // app.core.llm_segmentation / RenderedClip.segment_origin) -- a plain
  // heuristic candidate (the vast majority, and everything before this
  // feature existed) gets no badge here to avoid cluttering every card.
  const segmentBadge = clip.segment_origin === "llm"
    ? `<span class="pill pill-unknown" title="${escapeHtml(clip.segment_llm_reason || "Suggested by LLM analysis of the transcript.")}">🤖 LLM-suggested clip</span>`
    : "";
  // A stitched clip is joined from separate stretches of the VOD (see
  // CandidateSegment.parts) -- worth flagging explicitly, since the hard
  // cuts between parts are exactly the thing a reviewer should watch for
  // before approving one.
  const stitchedBadge = Array.isArray(clip.segment_parts) && clip.segment_parts.length > 1
    ? `<span class="pill pill-warn" title="${escapeHtml(
        clip.segment_parts.map(([s, e]) => `${Number(s).toFixed(1)}s-${Number(e).toFixed(1)}s`).join(" + ")
      )}">✂ Stitched from ${clip.segment_parts.length} parts</span>`
    : "";
  const explanationHtml = clip.caption_explanation
    ? `<div class="muted small">${escapeHtml(clip.caption_explanation)}</div>`
    : "";

  card.innerHTML = `
    <div class="row">
      <span class="pill ${statusClass(clip.status)}">${clip.status}</span>
      ${scoreBadge}
      ${sourceBadge}
      ${segmentBadge}
      ${stitchedBadge}
      <span class="mono muted">${clip.id}</span>
      <span class="muted">${clip.format}${durationText}${flagText}</span>
    </div>
    ${titleHtml}
    <div class="muted">${escapeHtml(clip.caption_text || "(no caption yet)")}</div>
    ${hashtagsHtml}
    ${explanationHtml}
    <div class="clip-preview"></div>
    <div class="row" style="margin-top:0.5rem">
      ${hasVideo
        ? '<button class="secondary preview-btn">▶ Watch clip</button><button class="secondary download-btn">⬇ Download clip</button>'
        : '<span class="muted">no rendered video yet</span>'}
      <button class="approve-btn">Approve</button>
      <button class="secondary reject-btn">Reject</button>
      <select class="account-select"><option value="">-- creator account --</option></select>
      <button class="secondary upload-btn">Upload to TikTok draft</button>
      <button class="secondary danger delete-clip-btn" title="Remove this clip from the review queue">🗑 Delete</button>
    </div>
    <div class="row rating-row" style="margin-top:0.35rem" title="Optional -- how good is this clip, separate from approve/reject? Helps sanity-check scoring weights later.">
      <span class="muted">Rate this clip:</span>
      <span class="rating-stars"></span>
      <span class="muted rating-clear-wrap" style="display:none">
        (<a href="#" class="rating-clear-link">clear</a>)
      </span>
    </div>
    <div class="review-notes-wrap" style="margin-top:0.35rem">
      <textarea class="review-notes-input" rows="2"
        placeholder="Why is this clip good or bad? (optional -- saved with Approve/Reject, and fed back to the clip finder as an example of what you like)"
      >${escapeHtml(clip.latest_notes || "")}</textarea>
    </div>
    <div class="muted review-result"></div>
  `;

  const select = card.querySelector(".account-select");
  for (const acct of creatorAccountsCache) {
    const opt = document.createElement("option");
    opt.value = acct.id;
    opt.textContent = `${acct.platform}:${acct.external_account_id}`;
    select.appendChild(opt);
  }

  const previewEl = card.querySelector(".clip-preview");
  if (clip.thumbnail_key) {
    loadClipThumbnail(clip.id, previewEl);
  }
  if (hasVideo) {
    card.querySelector(".preview-btn").addEventListener("click", () => loadClipVideo(clip.id, previewEl));
    card.querySelector(".download-btn").addEventListener("click", (e) => downloadClipVideo(clip, e.target));
  }
  // Separate from hasVideo: the thumbnail is its own artifact and is worth
  // offering on its own -- it is the cover image for a manual upload, and a
  // clip can have one even when the mp4 is gone.
  if (clip.thumbnail_key) {
    const thumbBtn = document.createElement("button");
    thumbBtn.className = "secondary download-thumb-btn";
    thumbBtn.textContent = "⬇ Thumbnail";
    thumbBtn.addEventListener("click", (e) => downloadClipThumbnail(clip, e.target));
    const row = card.querySelector(".preview-btn")?.parentElement
      || card.querySelector(".clip-preview")?.parentElement;
    if (row) row.appendChild(thumbBtn);
  }

  const resultEl = card.querySelector(".review-result");

  // Rating stars: a plain 1-5 click target, not a form control, so a
  // reviewer can change their mind before hitting Approve/Reject without
  // any extra "confirm" step -- the chosen value (or none) is read at
  // Approve/Reject time from card.dataset.selectedRating below.
  card.dataset.selectedRating = clip.latest_rating != null ? String(clip.latest_rating) : "";
  const starsEl = card.querySelector(".rating-stars");
  const clearWrapEl = card.querySelector(".rating-clear-wrap");
  function renderStars() {
    const current = card.dataset.selectedRating ? Number(card.dataset.selectedRating) : 0;
    starsEl.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const starBtn = document.createElement("button");
      starBtn.type = "button";
      starBtn.className = "star-btn";
      starBtn.textContent = i <= current ? "★" : "☆";
      starBtn.title = `Rate ${i}/5`;
      starBtn.addEventListener("click", () => {
        card.dataset.selectedRating = String(i);
        renderStars();
      });
      starsEl.appendChild(starBtn);
    }
    clearWrapEl.style.display = current > 0 ? "" : "none";
  }
  renderStars();
  card.querySelector(".rating-clear-link").addEventListener("click", (e) => {
    e.preventDefault();
    card.dataset.selectedRating = "";
    renderStars();
  });

  if (hasTitle) {
    const editBtn = card.querySelector(".edit-title-btn");
    const rowEl = card.querySelector(".clip-title-row");
    const formEl = card.querySelector(".title-edit-form");
    const inputEl = card.querySelector(".title-edit-input");
    editBtn.addEventListener("click", () => {
      inputEl.value = clip.caption_title;
      rowEl.style.display = "none";
      formEl.style.display = "";
      inputEl.focus();
    });
    card.querySelector(".cancel-title-btn").addEventListener("click", () => {
      formEl.style.display = "none";
      rowEl.style.display = "";
    });
    card.querySelector(".save-title-btn").addEventListener("click", () =>
      saveClipTitle(clip.id, inputEl.value, resultEl)
    );
  }

  if (hasHashtags) {
    const editBtn = card.querySelector(".edit-hashtags-btn");
    const rowEl = card.querySelector(".hashtags-row");
    const formEl = card.querySelector(".hashtag-edit-form");
    const inputEl = card.querySelector(".hashtag-edit-input");
    editBtn.addEventListener("click", () => {
      inputEl.value = clip.caption_hashtags.join(" ");
      rowEl.style.display = "none";
      formEl.style.display = "";
      inputEl.focus();
    });
    card.querySelector(".cancel-hashtags-btn").addEventListener("click", () => {
      formEl.style.display = "none";
      rowEl.style.display = "";
    });
    card.querySelector(".save-hashtags-btn").addEventListener("click", () =>
      saveClipHashtags(clip.id, inputEl.value, resultEl)
    );
  }

  const currentRating = () => (card.dataset.selectedRating ? Number(card.dataset.selectedRating) : null);
  // Free-text comment sent along with whichever decision button is
  // pressed -- empty box means "no comment on this review", not "erase
  // what I said before" (see RenderedClip.latest_notes).
  const notesEl = card.querySelector(".review-notes-input");
  const currentNotes = () => {
    const text = notesEl.value.trim();
    return text || null;
  };
  card.querySelector(".approve-btn").addEventListener("click", () =>
    reviewClip(clip.id, "approved", resultEl, currentRating(), currentNotes())
  );
  card.querySelector(".reject-btn").addEventListener("click", () =>
    reviewClip(clip.id, "rejected", resultEl, currentRating(), currentNotes())
  );
  card.querySelector(".upload-btn").addEventListener("click", () => {
    const accountId = select.value;
    if (!accountId) {
      showToast("Pick a creator account first");
      return;
    }
    uploadClip(clip.id, accountId, resultEl);
  });
  card.querySelector(".delete-clip-btn").addEventListener("click", () => deleteClip(clip.id));

  return card;
}

async function deleteClip(clipId) {
  if (!confirm("Delete this clip? This can't be undone.")) return;
  try {
    await apiFetch(`/api/v1/clips/${clipId}`, { method: "DELETE" });
  } catch {
    return; // toast already shown
  }
  lastRenderedClipsSignature = null; // force a fresh render so the deleted card disappears
  await loadJobDetail();
}

// Plain <video src="..."> / <img src="..."> can't carry the Authorization
// header the API requires, so preview media is fetched as an authenticated
// blob and shown via a local object URL instead -- same auth model as
// apiFetch, just not JSON. Thumbnails load eagerly (small); full video is
// behind a click so opening a job with many clips doesn't pull down every
// rendered mp4 at once.
async function fetchMediaBlobUrl(path) {
  const token = getToken();
  const resp = await fetch(`${API_BASE}${path}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  const blob = await resp.blob();
  return URL.createObjectURL(blob);
}

async function loadClipThumbnail(clipId, container) {
  if (container.querySelector("img, video")) return; // already loaded (or replaced by the video preview)
  try {
    const url = await fetchMediaBlobUrl(`/api/v1/clips/${clipId}/thumbnail`);
    container.innerHTML = `<img src="${url}" alt="clip thumbnail" style="max-width:180px;border-radius:8px;display:block" />`;
  } catch {
    /* no thumbnail yet -- not worth a toast, it's a nice-to-have */
  }
}

// TikTok overlays its own chrome on top of every video in the feed, so a
// clip that looks correctly framed in this preview can still have its
// captions, title or subject hidden behind the platform's UI. Percentages
// come from app.core.rendering_logic's TIKTOK_SAFE_* constants (researched
// 2026-09-11) -- keep the two in sync if those are ever re-measured.
const TIKTOK_SAFE_ZONES = {
  top: (200 / 1920) * 100,
  bottom: (334 / 1920) * 100,
  left: (86 / 1080) * 100,
  right: (140 / 1080) * 100,
};

async function loadClipVideo(clipId, container) {
  container.innerHTML = `<div class="muted">Loading preview…</div>`;
  try {
    const url = await fetchMediaBlobUrl(`/api/v1/clips/${clipId}/video`);
    const z = TIKTOK_SAFE_ZONES;
    container.innerHTML = `
      <div class="clip-preview">
        <div class="clip-preview-stage">
          <video controls src="${url}"></video>
          <div class="safe-zones" hidden>
            <div class="safe-zone safe-zone-top"    style="height:${z.top}%"></div>
            <div class="safe-zone safe-zone-bottom" style="height:${z.bottom}%"></div>
            <div class="safe-zone safe-zone-left"   style="width:${z.left}%"></div>
            <div class="safe-zone safe-zone-right"  style="width:${z.right}%"></div>
          </div>
        </div>
        <label class="safe-zone-toggle">
          <input type="checkbox" class="safe-zone-checkbox" />
          Show TikTok safe zones
          <span class="muted" title="Shaded areas are covered by TikTok's own UI in the For You feed: the top tabs/search, the bottom @username + caption + audio marquee, and the right-hand like/comment/share rail. Anything you put under the shading is rendered but not visible to a viewer.">?</span>
        </label>
      </div>`;
    const box = container.querySelector(".safe-zones");
    container.querySelector(".safe-zone-checkbox").addEventListener("change", (e) => {
      box.hidden = !e.target.checked;
    });
  } catch (err) {
    container.innerHTML = "";
    showToast(`Could not load clip preview: ${err.message}`);
  }
}

// Saves the clip's mp4 straight to the browser's downloads folder -- meant
// for manually posting to TikTok yourself while the real OAuth/API upload
// path (initConnectTikTokButton / uploadClip below) is still being set up,
// or any time you just want the file. Reuses fetchMediaBlobUrl's auth-blob
// trick since a plain <a href> can't carry the Authorization header; the
// download itself happens via a throwaway <a download> anchor, which is the
// standard way to save a blob without navigating the page away from it.
// Filename from the clip's own title, so a folder of downloads is readable
// instead of 24 UUIDs. Falls back to the id when there is no usable title.
function clipFilenameBase(clip) {
  return (
    (clip.caption_title || clip.id)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || clip.id
  );
}

// Shared by the mp4 and thumbnail buttons. Both need the same
// authenticated-blob dance: a plain <a href> can't carry the bearer header
// the API requires, so the bytes are fetched first and saved from a local
// object URL.
async function downloadClipMedia(clip, buttonEl, { path, extension, label }) {
  const originalText = buttonEl.textContent;
  buttonEl.textContent = "Downloading…";
  buttonEl.disabled = true;
  try {
    const url = await fetchMediaBlobUrl(`/api/v1/clips/${clip.id}/${path}`);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${clipFilenameBase(clip)}.${extension}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Release the blob URL once the download has had a moment to start --
    // revoking immediately can race the browser's own read of it.
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  } catch (err) {
    showToast(`Could not download ${label}: ${err.message}`);
  } finally {
    buttonEl.textContent = originalText;
    buttonEl.disabled = false;
  }
}

function downloadClipVideo(clip, buttonEl) {
  return downloadClipMedia(clip, buttonEl, { path: "video", extension: "mp4", label: "clip" });
}

function downloadClipThumbnail(clip, buttonEl) {
  return downloadClipMedia(clip, buttonEl, { path: "thumbnail", extension: "jpg", label: "thumbnail" });
}

async function reviewClip(clipId, decision, resultEl, rating = null, notes = null) {
  try {
    const result = await apiFetch(`/api/v1/clips/${clipId}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, rating, notes }),
    });
    const ratingText = result.rating != null ? `, rated ${result.rating}/5` : "";
    const notesText = result.notes ? ", comment saved" : "";
    resultEl.textContent =
      `Review saved: ${result.decision}${ratingText}${notesText} at ${new Date(result.decided_at).toLocaleTimeString()}`;
  } catch {
    /* toast already shown */
  }
}

async function saveClipTitle(clipId, rawInput, resultEl) {
  const title = rawInput.trim();
  if (!title) {
    showToast("Enter a title");
    return;
  }
  try {
    await apiFetch(`/api/v1/clips/${clipId}/caption`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    // Editing here only updates the stored record -- it does NOT
    // re-render the video, so it won't change an already-burned-in
    // banner (see ClipCaptionUpdate's docstring). Said plainly here so a
    // reviewer isn't surprised the clip itself doesn't change.
    resultEl.textContent =
      `Title updated at ${new Date().toLocaleTimeString()} (note: this doesn't change the video itself if it was already rendered)`;
    lastRenderedClipsSignature = null;
    await loadJobDetail();
  } catch {
    /* toast already shown */
  }
}

async function saveClipHashtags(clipId, rawInput, resultEl) {
  // Split on whitespace/commas -- accepts "#tag1 #tag2", "tag1, tag2",
  // whatever a reviewer naturally types. The API (app.core.caption_logic
  // .normalize_hashtags) re-adds the leading '#' and dedupes/caps the
  // count, so this doesn't need to be careful about exact formatting.
  const hashtags = rawInput.split(/[\s,]+/).map((t) => t.trim()).filter(Boolean);
  if (hashtags.length === 0) {
    showToast("Enter at least one hashtag");
    return;
  }
  try {
    await apiFetch(`/api/v1/clips/${clipId}/caption`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hashtags }),
    });
    resultEl.textContent = `Hashtags updated at ${new Date().toLocaleTimeString()}`;
    lastRenderedClipsSignature = null; // force the next poll/reload to rebuild the card with the new values
    await loadJobDetail();
  } catch {
    /* toast already shown */
  }
}

async function uploadClip(clipId, creatorAccountId, resultEl) {
  try {
    const result = await apiFetch(`/api/v1/clips/${clipId}/upload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ creator_account_id: creatorAccountId }),
    });
    resultEl.textContent = `Upload task ${result.id}: ${result.status}`;
  } catch {
    /* toast already shown */
  }
}

// ---- upload form ----

// Reads the optional upload-options fields into FormData entries the API's
// per-job overrides expect (app.api.routers.stream_jobs's Form(...) params)
// -- blank/unset means "don't send it", which the API treats as "use the
// global settings.* default", same as if this panel didn't exist.
function appendOptionalFormFields(form) {
  const stringField = (selector, name) => {
    const value = $(selector).value;
    if (value) form.append(name, value);
  };
  stringField("#opt-stt-model", "stt_model_size");
  stringField("#opt-max-clips", "max_clips");
  stringField("#opt-min-score", "min_score_threshold");
  stringField("#opt-min-clip-len", "min_clip_seconds");
  stringField("#opt-max-clip-len", "max_clip_seconds");
  stringField("#opt-camera-layout", "camera_layout_mode");
  stringField("#opt-crop-bias", "crop_bias");

  // Framing marked in the browser before upload. Multipart has no nested
  // objects, so each region goes as a JSON string; the API parses them with
  // the same model its JSON endpoint uses (see _parse_marked_rect).
  for (const [name, rect] of Object.entries(pfRegions)) {
    if (rect) form.append(`${name}_rect`, JSON.stringify(rect));
  }

  // Only send styling that actually differs from the server's defaults --
  // an untouched slider should leave the job on whatever .env says, not
  // pin it to today's default forever.
  const changed = {};
  for (const [k, v] of Object.entries(styleValues)) {
    if (v !== STYLE_DEFAULTS[k]) changed[k] = v;
  }
  if (Object.keys(changed).length) form.append('style_overrides', JSON.stringify(changed));
}

// ---- pre-upload framing picker ----
//
// The framing marks matter most BEFORE the pipeline runs, so this pulls a
// frame out of the chosen file locally -- a <video> fed an object URL, drawn
// to a <canvas> -- and never uploads or calls the API to do it. The marks
// ride along with the upload as form fields, so the first render is already
// framed correctly instead of the job having to exist first.
//
// Caveat this handles explicitly: browsers cannot decode every container we
// accept. .mkv in particular usually fails in Chrome even though ffmpeg on
// the server handles it fine. That is a preview limitation, not an upload
// limitation, so a failure here degrades to "upload anyway, mark it in Job
// detail" rather than blocking anything.
const PF_SETUPS_KEY = "clipping-machine.framing-setups";
const PF_CANVAS_WIDTH = 640;

let pfVideo = null;
let pfObjectUrl = null;
let pfMode = "facecam";
let pfRegions = { facecam: null, gameplay: null };
let pfReady = false;

function pfStatus(message) {
  const el = $("#pf-status");
  if (el) el.textContent = message;
}

function pfFormatTime(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

function pfDrawCurrentFrame() {
  if (!pfReady || !pfVideo) return;
  const canvas = $("#pf-canvas");
  const ratio = pfVideo.videoHeight / pfVideo.videoWidth || 9 / 16;
  canvas.width = PF_CANVAS_WIDTH;
  canvas.height = Math.round(PF_CANVAS_WIDTH * ratio);
  canvas.style.width = "100%";
  canvas.getContext("2d").drawImage(pfVideo, 0, 0, canvas.width, canvas.height);
  pfRenderRegions();
}

function pfRenderRegions() {
  const wrap = $("#pf-canvas-wrap");
  if (!wrap) return;
  wrap.querySelectorAll(".facecam-box.saved").forEach((el) => el.remove());
  const r = wrap.getBoundingClientRect();
  for (const [name, rect] of Object.entries(pfRegions)) {
    if (!rect) continue;
    const el = document.createElement("div");
    el.className = "facecam-box saved";
    const color = name === "facecam" ? "#4ade80" : "#60a5fa";
    el.style.borderColor = color;
    el.style.background = `${color}28`;
    el.style.left = `${rect.x * r.width}px`;
    el.style.top = `${rect.y * r.height}px`;
    el.style.width = `${rect.w * r.width}px`;
    el.style.height = `${rect.h * r.height}px`;
    const tag = document.createElement("span");
    tag.className = "facecam-box-tag";
    tag.style.background = color;
    tag.textContent = name;
    el.appendChild(tag);
    wrap.appendChild(el);
  }
}

function pfDescribe() {
  const bits = Object.entries(pfRegions)
    .filter(([, rect]) => rect)
    .map(([name, rect]) => `${name} ${(rect.w * 100).toFixed(0)}%x${(rect.h * 100).toFixed(0)}%`);
  return bits.length ? bits.join("  ·  ") : "nothing marked";
}

function pfLoadFile(file) {
  const panel = $("#pre-framing");
  if (!panel) return;
  panel.hidden = false;
  pfReady = false;
  pfRegions = { facecam: null, gameplay: null };
  pfRenderRegions();

  if (pfObjectUrl) URL.revokeObjectURL(pfObjectUrl);
  pfObjectUrl = URL.createObjectURL(file);

  if (!pfVideo) {
    pfVideo = document.createElement("video");
    pfVideo.muted = true;
    pfVideo.preload = "metadata";
    // Drawing a cross-origin frame would taint the canvas; an object URL of a
    // local file is same-origin, so this stays readable.
    pfVideo.addEventListener("seeked", pfDrawCurrentFrame);
  }

  pfStatus("Reading the file…");

  pfVideo.onloadedmetadata = () => {
    if (!pfVideo.videoWidth) {
      pfStatus("This browser can't decode this file for preview (common with .mkv). Upload anyway — you can mark framing in Job detail once the job exists.");
      return;
    }
    pfReady = true;
    const scrub = $("#pf-scrub");
    scrub.max = String(Math.max(1, Math.floor(pfVideo.duration || 1)));
    scrub.value = "0";
    $("#pf-time").textContent = pfFormatTime(0);
    // Not frame 0: the opening of a VOD is usually a starting-soon screen,
    // which shows nothing about the layout being marked.
    pfVideo.currentTime = Math.min(60, (pfVideo.duration || 2) / 2);
    pfStatus(`Scrub to a moment showing your layout, then drag a box around the ${pfMode}.`);
  };

  pfVideo.onerror = () => {
    pfStatus("This browser can't decode this file for preview (common with .mkv). Upload anyway — you can mark framing in Job detail once the job exists.");
  };

  pfVideo.src = pfObjectUrl;
}

function pfSetMode(mode) {
  pfMode = mode;
  $("#pf-mode-facecam").classList.toggle("active", mode === "facecam");
  $("#pf-mode-gameplay").classList.toggle("active", mode === "gameplay");
  pfStatus(`Drag a box around the ${mode}.  ${pfDescribe()}`);
}

function pfSavedSetups() {
  try {
    return JSON.parse(localStorage.getItem(PF_SETUPS_KEY) || "{}");
  } catch {
    return {}; // corrupt or storage disabled -- setups are a convenience, never required
  }
}

function pfRefreshSetupList() {
  const select = $("#pf-setups");
  if (!select) return;
  const setups = pfSavedSetups();
  select.innerHTML =
    '<option value="">Load a saved setup…</option>' +
    Object.keys(setups)
      .map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`)
      .join("");
}

function initPreFraming() {
  const wrap = $("#pf-canvas-wrap");
  if (!wrap) {
    console.warn("[clipping-machine] pre-upload framing markup missing -- hard-refresh (Ctrl+Shift+R)");
    return;
  }

  $("#file-input").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) pfLoadFile(file);
    else $("#pre-framing").hidden = true;
  });

  $("#pf-scrub").addEventListener("input", (e) => {
    if (!pfReady) return;
    const at = Number(e.target.value);
    $("#pf-time").textContent = pfFormatTime(at);
    pfVideo.currentTime = at; // 'seeked' redraws
  });

  $("#pf-mode-facecam").addEventListener("click", () => pfSetMode("facecam"));
  $("#pf-mode-gameplay").addEventListener("click", () => pfSetMode("gameplay"));

  $("#pf-clear").addEventListener("click", () => {
    pfRegions = { facecam: null, gameplay: null };
    pfRenderRegions();
    pfStatus("Boxes cleared — framing falls back to automatic detection.");
  });

  // --- drag to mark ---
  const live = $("#pf-live");
  let dragging = false;
  let originX = 0;
  let originY = 0;

  const relative = (event) => {
    const r = wrap.getBoundingClientRect();
    return {
      x: Math.min(Math.max(event.clientX - r.left, 0), r.width),
      y: Math.min(Math.max(event.clientY - r.top, 0), r.height),
      w: r.width,
      h: r.height,
    };
  };

  wrap.addEventListener("pointerdown", (e) => {
    if (!pfReady) return;
    const p = relative(e);
    dragging = true;
    originX = p.x;
    originY = p.y;
    wrap.setPointerCapture(e.pointerId);
    const color = pfMode === "facecam" ? "#4ade80" : "#60a5fa";
    live.hidden = false;
    live.style.borderColor = color;
    live.style.background = `${color}28`;
    live.style.left = `${p.x}px`;
    live.style.top = `${p.y}px`;
    live.style.width = "0px";
    live.style.height = "0px";
    e.preventDefault();
  });

  wrap.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const p = relative(e);
    live.style.left = `${Math.min(originX, p.x)}px`;
    live.style.top = `${Math.min(originY, p.y)}px`;
    live.style.width = `${Math.abs(p.x - originX)}px`;
    live.style.height = `${Math.abs(p.y - originY)}px`;
  });

  const finish = (e) => {
    if (!dragging) return;
    dragging = false;
    live.hidden = true;
    const p = relative(e);
    const w = Math.abs(p.x - originX);
    const h = Math.abs(p.y - originY);
    if (w < 8 || h < 8) {
      pfStatus(`That box was too small to count. Drag a box around the ${pfMode}.`);
      return;
    }
    pfRegions[pfMode] = {
      x: Math.min(originX, p.x) / p.w,
      y: Math.min(originY, p.y) / p.h,
      w: w / p.w,
      h: h / p.h,
    };
    pfRenderRegions();
    pfStatus(`${pfDescribe()} — these upload with the job.`);
  };

  wrap.addEventListener("pointerup", finish);
  wrap.addEventListener("pointercancel", finish);
  window.addEventListener("resize", pfRenderRegions);

  // --- saved setups (your rig is the same every stream) ---
  $("#pf-save-setup").addEventListener("click", () => {
    const name = $("#pf-setup-name").value.trim();
    if (!name) return showToast("Give the setup a name first");
    if (!pfRegions.facecam && !pfRegions.gameplay) return showToast("Mark at least one box first");
    try {
      const setups = pfSavedSetups();
      setups[name] = pfRegions;
      localStorage.setItem(PF_SETUPS_KEY, JSON.stringify(setups));
      pfRefreshSetupList();
      pfStatus(`Saved "${name}". Pick it from the list on your next upload.`);
    } catch {
      showToast("Could not save the setup (browser storage unavailable)");
    }
  });

  $("#pf-setups").addEventListener("change", (e) => {
    const name = e.target.value;
    if (!name) return;
    const setups = pfSavedSetups();
    if (!setups[name]) return;
    pfRegions = { facecam: setups[name].facecam || null, gameplay: setups[name].gameplay || null };
    pfRenderRegions();
    pfStatus(`Loaded "${name}" — ${pfDescribe()}.`);
  });

  pfRefreshSetupList();
}

// ---- layout & text preview ----
//
// The point of this panel is that it is FAITHFUL, not decorative. Every
// number below came from rendering real libass output on a 1080x1920 frame
// and measuring the lit pixel rows -- libass interprets FontSize against its
// own script resolution, so nothing here can be derived from the ASS spec.
//
//   caption clearance from the bottom :  28->188px  40->268  55->368  70->468
//   title top edge from the top       :  10->80px   30->214  35->247  50->347
//   caption ink height                :  fs8->45px  fs9->50  fs10->56
//
// Each is linear over the usable range, so a slope+intercept reproduces the
// real geometry closely enough that the phone matches the render.
const LAYOUT_METRICS = {
  OUT_W: 1080,
  OUT_H: 1920,
  captionClearancePx: (marginV) => 6.67 * marginV,
  titleTopPx: (marginV) => 6.675 * marginV + 13,
  captionInkPx: (fontSize) => 5.5 * fontSize + 1,
  // The title is bold with a heavier outline, so it does not share the
  // caption's slope: measured 246px for two lines at size 13 => ~123 each.
  titleInkPx: (fontSize) => 9.46 * fontSize,
  // TikTok's own chrome, same constants as app/core/rendering_logic.py.
  safe: { top: 200, bottom: 334, left: 86, right: 140 },
};

const STYLE_DEFAULTS = {
  split_facecam_fraction: 0.35,
  caption_font_size: 8,
  caption_margin_v: 55,
  caption_max_chars: 80,
  title_font_size: 13,
  title_margin_v: 30,
};

const STYLE_KEY = "clipping-machine.style-overrides";
let styleValues = { ...STYLE_DEFAULTS };

function loadSavedStyle() {
  try {
    const raw = JSON.parse(localStorage.getItem(STYLE_KEY) || "{}");
    for (const k of Object.keys(STYLE_DEFAULTS)) {
      if (typeof raw[k] === "number") styleValues[k] = raw[k];
    }
  } catch {
    /* corrupt or storage disabled -- defaults are fine */
  }
}

function saveStyle() {
  try {
    localStorage.setItem(STYLE_KEY, JSON.stringify(styleValues));
  } catch {
    /* a convenience, never required */
  }
}

// CSS font-size is an em box; the measurements above are INK height (the lit
// rows). For typical faces the ink of a mixed-case line is ~75% of the em,
// so dividing keeps the preview text visually the same size as the render.
function inkToCssPx(inkPx, scale) {
  return (inkPx * scale) / 0.75;
}

function renderPhonePreview() {
  const phone = $("#phone");
  if (!phone) return;
  const M = LAYOUT_METRICS;
  const h = phone.clientHeight || 480;
  const scale = h / M.OUT_H;

  const camPct = styleValues.split_facecam_fraction * 100;
  $("#ph-cam").style.height = `${camPct}%`;
  $("#ph-game").style.height = `${100 - camPct}%`;

  const titleTop = M.titleTopPx(styleValues.title_margin_v);
  const titleEl = $("#ph-title");
  titleEl.style.top = `${titleTop * scale}px`;
  titleEl.style.fontSize = `${inkToCssPx(M.titleInkPx(styleValues.title_font_size), scale)}px`;

  const capClear = M.captionClearancePx(styleValues.caption_margin_v);
  const capEl = $("#ph-caption");
  capEl.style.bottom = `${capClear * scale}px`;
  capEl.style.fontSize = `${inkToCssPx(M.captionInkPx(styleValues.caption_font_size), scale)}px`;

  // safe zones, as a share of the output
  const z = $("#ph-zones");
  z.querySelector(".safe-zone-top").style.height = `${(M.safe.top / M.OUT_H) * 100}%`;
  z.querySelector(".safe-zone-bottom").style.height = `${(M.safe.bottom / M.OUT_H) * 100}%`;
  z.querySelector(".safe-zone-left").style.width = `${(M.safe.left / M.OUT_W) * 100}%`;
  z.querySelector(".safe-zone-right").style.width = `${(M.safe.right / M.OUT_W) * 100}%`;

  // readouts
  const camPx = Math.floor(M.OUT_H * styleValues.split_facecam_fraction / 2) * 2;
  $("#out-split").textContent =
    `${Math.round(camPct)}% cam — ${camPx}px cam / ${M.OUT_H - camPx}px gameplay`;
  $("#out-capsize").textContent =
    `${styleValues.caption_font_size} — about ${Math.round(1080 / (M.captionInkPx(styleValues.caption_font_size) * 0.62))} chars per line`;
  $("#out-capmargin").textContent = `${styleValues.caption_margin_v} — ${Math.round(capClear)}px above the bottom`;
  $("#out-capchars").textContent = `${styleValues.caption_max_chars} characters`;
  $("#out-titlesize").textContent = `${styleValues.title_font_size}`;
  $("#out-titlemargin").textContent = `${styleValues.title_margin_v} — starts ${Math.round(titleTop)}px down`;

  // Warnings, because the sliders let you go somewhere the render will look
  // fine in this console and be half-hidden in the actual feed.
  const warnings = [];
  if (capClear < M.safe.bottom) {
    warnings.push(`Caption sits ${Math.round(M.safe.bottom - capClear)}px inside TikTok's bottom UI — it will be covered by the username and caption text.`);
  }
  if (titleTop < M.safe.top) {
    warnings.push(`Title starts ${Math.round(M.safe.top - titleTop)}px behind TikTok's top navigation.`);
  }
  const warnEl = $("#sty-warnings");
  warnEl.innerHTML = warnings.length
    ? warnings.map((w) => `<div style="color:var(--bad)">⚠ ${escapeHtml(w)}</div>`).join("")
    : `<div style="color:var(--ok)">✓ Title and captions both clear TikTok's UI.</div>`;
}

const STYLE_SLIDERS = [
  ["#sty-split", "split_facecam_fraction", parseFloat],
  ["#sty-capsize", "caption_font_size", parseInt],
  ["#sty-capmargin", "caption_margin_v", parseInt],
  ["#sty-capchars", "caption_max_chars", parseInt],
  ["#sty-titlesize", "title_font_size", parseInt],
  ["#sty-titlemargin", "title_margin_v", parseInt],
];

function syncSlidersFromValues() {
  for (const [sel, key] of STYLE_SLIDERS) {
    const el = $(sel);
    if (el) el.value = String(styleValues[key]);
  }
  renderPhonePreview();
}

function initLayoutEditor() {
  if (!$("#phone")) {
    console.warn("[clipping-machine] layout editor markup missing -- hard-refresh (Ctrl+Shift+R)");
    return;
  }
  loadSavedStyle();

  for (const [sel, key, parse] of STYLE_SLIDERS) {
    $(sel).addEventListener("input", (e) => {
      styleValues[key] = parse(e.target.value);
      saveStyle();
      renderPhonePreview();
    });
  }

  $("#sty-reset").addEventListener("click", () => {
    styleValues = { ...STYLE_DEFAULTS };
    saveStyle();
    syncSlidersFromValues();
  });

  $("#ph-show-zones").addEventListener("change", (e) => {
    $("#ph-zones").hidden = !e.target.checked;
  });

  window.addEventListener("resize", renderPhonePreview);
  syncSlidersFromValues();
}

// ---- AI model settings ----
//
// The API key never comes back from the server -- GET reports only whether
// one is set and its last four characters. So the input is always blank on
// load, and an empty input on save means "leave it alone" rather than
// "delete it"; deletion is its own explicit button. That asymmetry is
// deliberate: a blank field silently wiping a working key is exactly the
// kind of thing you only notice a render later.
let llmSettings = null;

function renderLlmSettings() {
  if (!llmSettings) return;
  const s = llmSettings;
  const chosen = s.provider || s.effective_provider;
  $("#llm-ollama").checked = chosen === "ollama";
  $("#llm-openai").checked = chosen === "openai";

  $("#llm-ollama-desc").textContent = `free, private, slower — ${s.ollama_model}`;
  $("#llm-openai-desc").textContent = `better selection, costs per use — ${s.openai_model}`;

  $("#llm-key-state").textContent = s.has_own_key
    ? `saved (${s.key_hint})`
    : s.server_has_key
      ? "none saved — the server's own key would be used"
      : "none saved";

  const notes = [];
  if (!s.segment_suggestions_enabled) {
    notes.push(
      "Transcript analysis is currently OFF (ENABLE_LLM_SEGMENT_SUGGESTIONS in .env), so no model " +
      "is reading your transcript to find clip-worthy moments regardless of what you pick here."
    );
  }
  if (s.effective_provider === "openai" && !s.has_own_key && s.server_has_key) {
    notes.push("Using the server's .env key, not one of yours.");
  }
  notes.push("Still set in .env, not here: model names, the Ollama URL, and whether transcript analysis runs at all.");
  $("#llm-note").innerHTML = notes.map((n) => escapeHtml(n)).join("<br>");
}

async function loadLlmSettings() {
  try {
    llmSettings = await apiFetch("/api/v1/settings/llm");
    renderLlmSettings();
  } catch {
    /* toast already shown */
  }
}

async function saveLlmSettings({ removeKey = false } = {}) {
  const body = { provider: $("#llm-openai").checked ? "openai" : "ollama" };
  const typed = $("#llm-key").value.trim();
  if (removeKey) {
    body.api_key = null;
  } else if (typed) {
    body.api_key = typed;
  }
  // else: api_key omitted entirely -> server leaves the stored key untouched

  const status = $("#llm-status");
  status.textContent = "Saving…";
  try {
    llmSettings = await apiFetch("/api/v1/settings/llm", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    $("#llm-key").value = "";
    renderLlmSettings();
    status.textContent = removeKey ? "Key removed." : "Saved.";
    setTimeout(() => { status.textContent = ""; }, 4000);
  } catch {
    status.textContent = "";
  }
}

function initLlmSettings() {
  if (!$("#llm-save")) {
    console.warn("[clipping-machine] AI model panel missing -- hard-refresh (Ctrl+Shift+R)");
    return;
  }
  $("#llm-save").addEventListener("click", () => saveLlmSettings());
  $("#llm-remove-key").addEventListener("click", () => {
    if (confirm("Remove the saved API key? Jobs fall back to the server's key, or to Ollama.")) {
      saveLlmSettings({ removeKey: true });
    }
  });
}

function initUploadForm() {
  $("#upload-btn").addEventListener("click", async () => {
    const fileInput = $("#file-input");
    const status = $("#upload-status");
    if (!fileInput.files.length) {
      showToast("Choose a file first");
      return;
    }
    const form = new FormData();
    form.append("file", fileInput.files[0]);
    appendOptionalFormFields(form);
    status.textContent = "Uploading…";
    try {
      const job = await apiFetch("/api/v1/stream-jobs", { method: "POST", body: form });
      status.textContent = `Created job ${job.id} (status: ${job.status})`;
      fileInput.value = "";
      await loadJobs();
      await selectJob(job.id);
    } catch {
      status.textContent = "Upload failed -- see error above.";
    }
  });
}

// ---- creator accounts ----

async function loadAccounts() {
  let accounts;
  try {
    accounts = await apiFetch("/api/v1/creator-accounts");
  } catch {
    return;
  }
  creatorAccountsCache = accounts;
  const body = $("#accounts-body");
  body.innerHTML = "";
  for (const acct of accounts) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${acct.platform}</td>
      <td>${escapeHtml(acct.external_account_id)}</td>
      <td>${acct.daily_upload_cap}</td>
      <td class="mono muted">${acct.id}</td>
    `;
    body.appendChild(tr);
  }
}

function initAddAccountForm() {
  $("#add-account-btn").addEventListener("click", async () => {
    const externalId = $("#acct-external-id").value.trim();
    const accessToken = $("#acct-access-token").value.trim();
    if (!externalId || !accessToken) {
      showToast("External account id and access token are required");
      return;
    }
    const body = {
      platform: $("#acct-platform").value,
      external_account_id: externalId,
      access_token: accessToken,
    };
    const refreshToken = $("#acct-refresh-token").value.trim();
    if (refreshToken) body.refresh_token = refreshToken;
    const dailyCap = $("#acct-daily-cap").value.trim();
    if (dailyCap) body.daily_upload_cap = Number(dailyCap);

    try {
      await apiFetch("/api/v1/creator-accounts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch {
      return; // toast already shown
    }
    // Clear the token fields (not the platform picker) so a mistyped token
    // isn't left sitting in the form after a successful add.
    $("#acct-external-id").value = "";
    $("#acct-access-token").value = "";
    $("#acct-refresh-token").value = "";
    $("#acct-daily-cap").value = "";
    showToast("Creator account added");
    await loadAccounts();
  });
}

function initConnectTikTokButton() {
  $("#connect-tiktok-btn").addEventListener("click", async () => {
    let resp;
    try {
      resp = await apiFetch("/api/v1/creator-accounts/tiktok/oauth/start");
    } catch {
      // apiFetch already toasted the error (e.g. tiktok_not_configured if
      // TIKTOK_CLIENT_KEY/TIKTOK_REDIRECT_URI aren't set -- see README).
      return;
    }
    // Full-page redirect to TikTok's own consent screen -- not a fetch,
    // since the whole point is leaving this page. TikTok redirects back to
    // TIKTOK_REDIRECT_URI (the backend's own /tiktok/oauth/callback route,
    // a server-rendered HTML page, not part of this SPA) once the creator
    // approves or declines.
    window.location.href = resp.authorize_url;
  });
}

// ---- refresh loop ----

async function refreshAll() {
  await checkHealth();
  if (!currentUser) return; // not signed in yet -- don't fire 401s during startup/redirect
  await loadJobs();
  await loadAccounts();
  if (selectedJobId) await loadJobDetail();
}

function initAutoRefresh() {
  const checkbox = $("#auto-refresh");
  function apply() {
    if (autoRefreshTimer) clearInterval(autoRefreshTimer);
    if (checkbox.checked) {
      autoRefreshTimer = setInterval(refreshAll, 3000);
    }
  }
  checkbox.addEventListener("change", apply);
  apply();
}

// ---- init ----

document.addEventListener("DOMContentLoaded", async () => {
  $("#api-base-display").textContent = API_BASE;
  initSessionUi();

  // Everything below needs a session. requireAuth() redirects to the login
  // page on 401, so bailing here is what stops a burst of doomed requests
  // firing while the browser is already navigating away.
  if (!(await requireAuth())) return;
  initUploadForm();
  initFacecamMarker();
  initPreFraming();
  initLayoutEditor();
  initLlmSettings();
  initAddAccountForm();
  initConnectTikTokButton();
  $("#refresh-jobs-btn").addEventListener("click", loadJobs);
  $("#refresh-accounts-btn").addEventListener("click", loadAccounts);
  initAutoRefresh();
  refreshAll();
  loadLlmSettings();
});
