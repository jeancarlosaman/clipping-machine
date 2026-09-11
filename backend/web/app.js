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

// ---- token handling ----

function getToken() {
  return localStorage.getItem(TOKEN_KEY) || "";
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
  $("#token-status").textContent = token ? "saved" : "";
}

function initTokenPanel() {
  $("#token-input").value = getToken();
  $("#token-status").textContent = getToken() ? "saved" : "";
  $("#token-save").addEventListener("click", () => {
    setToken($("#token-input").value.trim());
    refreshAll();
  });
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
        <button class="secondary danger delete-job-btn" data-job-id="${job.id}" title="Delete this upload and its clips">🗑</button>
      </td>
    `;
    tr.querySelector(".view-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      selectJob(job.id);
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

// ---- facecam marker ----
//
// Face detection answering "is there a facecam, and where" is what produced
// the bad framing this exists to replace: it split clips on VODs with no
// facecam at all, and missed real ones. A creator looking at a frame of
// their own VOD is the one input that cannot be wrong, so when a box is
// marked here it overrides detection entirely (see StreamJob.facecam_rect
// and app/workers/rendering.py's layout decision).
//
// Coordinates are sent NORMALIZED (0..1 fractions of the frame), computed
// against the image's own displayed size -- so the browser never needs to
// know the source resolution and the mark stays correct if the same box is
// later applied to a different-resolution source.
let facecamDraft = null; // {x,y,w,h} normalized, pending save

function facecamSetStatus(message, isError = false) {
  const el = $("#facecam-status");
  if (!el) return;
  el.textContent = message;
  el.style.color = isError ? "var(--danger, #d66)" : "";
}

async function loadFacecamFrame() {
  if (!selectedJobId) return;
  const stage = $("#facecam-stage");
  const at = Number($("#facecam-at").value || 0);
  stage.innerHTML = `<div class="muted">Loading frame…</div>`;
  facecamDraft = null;
  $("#facecam-save").disabled = true;

  let url;
  try {
    url = await fetchMediaBlobUrl(`/api/v1/stream-jobs/${selectedJobId}/frame?at_seconds=${at}`);
  } catch (err) {
    stage.innerHTML = "";
    facecamSetStatus(
      `Could not load a frame at ${at}s (${err.message}). If the job is still uploading or that ` +
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

  // Rect is read per-drag rather than cached: the image can reflow (window
  // resize, the <details> being reopened) between drags.
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
    box.style.left = `${x}px`;
    box.style.top = `${y}px`;
    box.style.width = `${w}px`;
    box.style.height = `${h}px`;
  };

  canvas.addEventListener("pointerdown", (e) => {
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
    // A stray click is a 0-size box, not an instruction -- ignore it rather
    // than saving something that would fail validation server-side.
    if (w < 8 || h < 8) {
      box.hidden = true;
      facecamDraft = null;
      $("#facecam-save").disabled = true;
      facecamSetStatus("Drag a box around the facecam (that one was too small to count).");
      return;
    }
    facecamDraft = { x: x / p.w, y: y / p.h, w: w / p.w, h: h / p.h };
    $("#facecam-save").disabled = false;
    facecamSetStatus(
      `Box: ${(facecamDraft.w * 100).toFixed(1)}% x ${(facecamDraft.h * 100).toFixed(1)}% of the frame. ` +
      `Press "Save box" to use it.`,
    );
  };

  canvas.addEventListener("pointerup", finishDrag);
  canvas.addEventListener("pointercancel", finishDrag);

  facecamSetStatus("Drag a box around the facecam.");
}

async function saveFacecamRect(rect) {
  if (!selectedJobId) return;
  try {
    await apiFetch(`/api/v1/stream-jobs/${selectedJobId}/facecam-rect`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rect }),
    });
    facecamSetStatus(
      rect
        ? "Saved. The next render of this job's clips will use this box (already-rendered clips keep their framing)."
        : "Mark cleared -- this job is back to automatic face detection.",
    );
    await loadJobDetail();
  } catch {
    /* toast already shown */
  }
}

function initFacecamMarker() {
  $("#facecam-load").addEventListener("click", loadFacecamFrame);
  $("#facecam-save").addEventListener("click", () => {
    if (facecamDraft) saveFacecamRect(facecamDraft);
  });
  $("#facecam-clear").addEventListener("click", () => {
    facecamDraft = null;
    $("#facecam-save").disabled = true;
    const box = $("#facecam-box");
    if (box) box.hidden = true;
    saveFacecamRect(null);
  });
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

  const marker = $("#facecam-marker");
  if (marker) {
    const r = job.facecam_rect;
    marker.querySelector("summary").textContent = r
      ? `Facecam marked (${(r.w * 100).toFixed(0)}% x ${(r.h * 100).toFixed(0)}% of the frame) -- click to change`
      : "Mark the facecam (skip guessing where it is)";
  }

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
        ? '<button class="secondary preview-btn">▶ Watch clip</button><button class="secondary download-btn">⬇ Download</button>'
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
async function downloadClipVideo(clip, buttonEl) {
  const originalText = buttonEl.textContent;
  buttonEl.textContent = "Downloading…";
  buttonEl.disabled = true;
  try {
    const url = await fetchMediaBlobUrl(`/api/v1/clips/${clip.id}/video`);
    const filenameBase = (clip.caption_title || clip.id)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 60) || clip.id;
    const a = document.createElement("a");
    a.href = url;
    a.download = `${filenameBase}.mp4`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    // Release the blob URL once the download has had a moment to start --
    // revoking immediately can race the browser's own read of it.
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  } catch (err) {
    showToast(`Could not download clip: ${err.message}`);
  } finally {
    buttonEl.textContent = originalText;
    buttonEl.disabled = false;
  }
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
  if (!getToken()) return; // avoid firing (and console-logging) 401s before a token is set
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

document.addEventListener("DOMContentLoaded", () => {
  $("#api-base-display").textContent = API_BASE;
  initTokenPanel();
  initUploadForm();
  initFacecamMarker();
  initAddAccountForm();
  initConnectTikTokButton();
  $("#refresh-jobs-btn").addEventListener("click", loadJobs);
  $("#refresh-accounts-btn").addEventListener("click", loadAccounts);
  initAutoRefresh();
  refreshAll();
});
