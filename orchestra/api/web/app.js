// orchestra-rag web UI — live multi-agent collaboration over SSE.
"use strict";

const $ = (id) => document.getElementById(id);

const els = {
  statusDot: $("statusDot"),
  statusText: $("statusText"),
  kbCount: $("kbCount"),
  dropZone: $("dropZone"),
  fileInput: $("fileInput"),
  uploadBtn: $("uploadBtn"),
  uploadResult: $("uploadResult"),
  backend: $("backend"),
  strategy: $("strategy"),
  topk: $("topk"),
  question: $("question"),
  askBtn: $("askBtn"),
  timeline: $("timeline"),
  emptyState: $("emptyState"),
  answer: $("answer"),
  citations: $("citations"),
};

let lastCitations = [];

// ---- health / KB size -----------------------------------------------------
async function refreshHealth() {
  try {
    const r = await fetch("/health");
    const data = await r.json();
    els.statusDot.classList.add("ok");
    els.statusText.textContent = `ready · ${data.backend}`;
    els.kbCount.textContent = `${data.chunks} chunks indexed`;
  } catch (e) {
    els.statusDot.classList.add("err");
    els.statusText.textContent = "offline";
  }
}

// ---- upload ---------------------------------------------------------------
function wireUpload() {
  els.dropZone.addEventListener("click", () => els.fileInput.click());
  ["dragover", "dragenter"].forEach((ev) =>
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      els.dropZone.classList.add("drag");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      els.dropZone.classList.remove("drag");
    })
  );
  els.dropZone.addEventListener("drop", (e) => {
    if (e.dataTransfer.files.length) {
      els.fileInput.files = e.dataTransfer.files;
      els.uploadResult.textContent = `${e.dataTransfer.files.length} file(s) selected`;
    }
  });
  els.fileInput.addEventListener("change", () => {
    if (els.fileInput.files.length) {
      els.uploadResult.textContent = `${els.fileInput.files.length} file(s) selected`;
    }
  });
  els.uploadBtn.addEventListener("click", doUpload);
}

async function doUpload() {
  if (!els.fileInput.files.length) {
    els.uploadResult.textContent = "Choose at least one file first.";
    return;
  }
  const fd = new FormData();
  for (const f of els.fileInput.files) fd.append("files", f);
  els.uploadBtn.disabled = true;
  els.uploadResult.textContent = "Ingesting…";
  try {
    const r = await fetch("/ingest", { method: "POST", body: fd });
    const data = await r.json();
    els.uploadResult.textContent = `+${data.ingested_chunks} chunks · ${data.total_chunks} total`;
    els.kbCount.textContent = `${data.total_chunks} chunks indexed`;
  } catch (e) {
    els.uploadResult.textContent = "Upload failed.";
  } finally {
    els.uploadBtn.disabled = false;
  }
}

// ---- timeline rendering ---------------------------------------------------
function clearTimeline() {
  els.timeline.innerHTML = "";
  els.emptyState = null;
}

function addRoundSep(n) {
  const div = document.createElement("div");
  div.className = "round-sep";
  div.textContent = `— round ${n} —`;
  els.timeline.appendChild(div);
}

function addThinking(role) {
  const card = document.createElement("div");
  card.className = `card thinking ${role}`;
  card.dataset.thinkingRole = role;
  card.innerHTML = `
    <div class="head"><span class="role">${role}</span></div>
    <div class="body">working…</div>`;
  els.timeline.appendChild(card);
  els.timeline.scrollTop = els.timeline.scrollHeight;
  return card;
}

function addMessage(role, content, round, approved, numPassages) {
  // Replace the matching "thinking" placeholder if present.
  const placeholder = els.timeline.querySelector(
    `.card.thinking[data-thinking-role="${role}"]`
  );
  const card = placeholder || document.createElement("div");
  card.className = `card ${role}`;
  delete card.dataset.thinkingRole;

  let badge = "";
  if (approved) badge = `<span class="badge">approved</span>`;
  else if (typeof numPassages === "number")
    badge = `<span class="badge">${numPassages} passages</span>`;

  const roundLabel = round ? `<span class="round">round ${round}</span>` : "";
  card.innerHTML = `
    <div class="head">
      <span class="role">${role}</span>
      ${badge}
      ${roundLabel}
    </div>
    <div class="body">${escapeHtml(content)}</div>`;
  if (!placeholder) els.timeline.appendChild(card);
  els.timeline.scrollTop = els.timeline.scrollHeight;
}

function escapeHtml(s) {
  return (s || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ---- answer + citations ---------------------------------------------------
function renderAnswer(text) {
  // Turn [n] into clickable citation chips that flash the source card.
  const html = escapeHtml(text).replace(
    /\[(\d+)\]/g,
    '<span class="cite" data-n="$1">[$1]</span>'
  );
  els.answer.innerHTML = html;
  els.answer.querySelectorAll(".cite").forEach((el) => {
    el.addEventListener("click", () => flashCitation(el.dataset.n));
  });
}

function renderCitations(citations) {
  lastCitations = citations || [];
  if (!lastCitations.length) {
    els.citations.innerHTML = `<span class="muted">No citations.</span>`;
    return;
  }
  els.citations.innerHTML = "";
  for (const c of lastCitations) {
    const div = document.createElement("div");
    div.className = "cite-card";
    div.id = `cite-${c.n}`;
    div.innerHTML = `
      <div><span class="cn">[${c.n}]</span><span class="src">${escapeHtml(
      c.source
    )} · score ${c.score}</span></div>
      <div class="txt">${escapeHtml(c.text)}</div>`;
    els.citations.appendChild(div);
  }
}

function flashCitation(n) {
  const card = $(`cite-${n}`);
  if (!card) return;
  card.classList.add("flash");
  card.scrollIntoView({ behavior: "smooth", block: "nearest" });
  setTimeout(() => card.classList.remove("flash"), 1200);
}

// ---- ask (SSE) ------------------------------------------------------------
async function ask() {
  const question = els.question.value.trim();
  if (!question) return;

  els.askBtn.disabled = true;
  clearTimeline();
  els.answer.innerHTML = `<span class="muted">agents are collaborating…</span>`;
  els.citations.innerHTML = `<span class="muted">—</span>`;

  const body = {
    question,
    backend: els.backend.value,
    strategy: els.strategy.value,
    k: parseInt(els.topk.value, 10) || 4,
  };

  try {
    const resp = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      els.answer.innerHTML = `<span class="muted">Error: ${escapeHtml(
        err.detail || "request failed"
      )}</span>`;
      return;
    }
    await consumeStream(resp.body.getReader());
  } catch (e) {
    els.answer.innerHTML = `<span class="muted">Error: ${escapeHtml(
      e.message
    )}</span>`;
  } finally {
    els.askBtn.disabled = false;
    refreshHealth();
  }
}

async function consumeStream(reader) {
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop(); // keep the partial frame
    for (const frame of parts) {
      const line = frame.split("\n").find((l) => l.startsWith("data:"));
      if (!line) continue;
      let evt;
      try {
        evt = JSON.parse(line.slice(5).trim());
      } catch {
        continue;
      }
      handleEvent(evt);
    }
  }
}

function handleEvent(evt) {
  switch (evt.type) {
    case "start":
      break;
    case "round":
      addRoundSep(evt.round);
      break;
    case "agent_start":
      addThinking(evt.role);
      break;
    case "agent_message":
      addMessage(
        evt.role,
        evt.content,
        evt.round,
        evt.approved,
        evt.num_passages
      );
      break;
    case "final":
      renderAnswer(evt.content);
      renderCitations(evt.citations);
      break;
    case "error":
      els.answer.innerHTML = `<span class="muted">Error: ${escapeHtml(
        evt.content
      )}</span>`;
      break;
    case "done":
      break;
  }
}

// ---- boot -----------------------------------------------------------------
wireUpload();
els.askBtn.addEventListener("click", ask);
els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter") ask();
});
refreshHealth();
