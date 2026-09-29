import { api, errorMessage } from "./api.js";

const $ = (sel) => document.querySelector(sel);
const state = {
  page: "ask",
  settings: null,
  form: null,
  draftDirty: false,
  documents: [],
  conversations: [],
  conversationId: null,
  messages: [],
  turnIndex: null,
  scopeId: null,
  analyticsDays: 30,
  docQuery: "",
  docFilter: "all",
  index: null,
  busy: false,
  uploading: false,
  replaceId: null,
  activeDocumentId: null,
};

const pageLabels = {
  ask: "Ask",
  knowledge: "Knowledge base",
  document: "Document",
  pipeline: "Index pipeline",
  analytics: "Analytics",
  settings: "Settings",
};

const pipeArrow = `<div class="pipe-arrow" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14m-5-5 5 5-5 5"/></svg></div>`;

const notify = (message) => {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => toast.classList.remove("show"), 2800);
};

const listen = (node, type, fn) => {
  if (!node) return;
  node.addEventListener(type, (event) => {
    Promise.resolve(fn(event)).catch((err) => notify(err.message || "Request failed"));
  });
};

const escapeHtml = (value) =>
  String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const initials = (name) =>
  String(name || "OP")
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || "")
    .join("") || "OP";

const bytes = (size) => {
  const n = Number(size) || 0;
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

const when = (iso) => {
  if (!iso) return "Just now";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "Just now";
  const diff = Date.now() - date.getTime();
  const mins = Math.round(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins} min ago`;
  const hours = Math.round(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return date.toLocaleDateString();
};

const fileKind = (name) => {
  const ext = String(name || "").split(".").pop().toLowerCase();
  if (ext === "pdf") return "PDF";
  if (ext === "docx") return "DOC";
  if (ext === "csv" || ext === "xlsx") return "CSV";
  if (ext === "pptx") return "PPT";
  return "TXT";
};

const percent = (value) => `${Math.round(Number(value || 0) * 1000) / 10}%`;

const optionList = (values, current) => {
  const list = [...values];
  if (current && !list.includes(current)) list.unshift(current);
  return list.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
};

const uniqueById = (items) => {
  const seen = new Set();
  return (items || []).filter((item) => {
    if (!item?.id || seen.has(item.id)) return false;
    seen.add(item.id);
    return true;
  });
};

const pendingCopy = {
  queued: "This document is still queued.",
  processing: "This document is still processing.",
  failed: "This document could not be indexed.",
};

function setSidebar(open) {
  $("#sidebar").classList.toggle("open", open);
  $("#threadsToggle").setAttribute("aria-expanded", open ? "true" : "false");
}

function setInspector(open) {
  $("#inspector").classList.toggle("open", open);
  $("#sourceToggle").setAttribute("aria-expanded", open ? "true" : "false");
}

function showPage(page) {
  state.page = page;
  $("#app").classList.toggle("subpage", page !== "ask");
  document.querySelectorAll(".workspace-page").forEach((view) => view.classList.toggle("active", view.dataset.view === page));
  document.querySelectorAll(".rail [data-page]").forEach((button) => {
    button.classList.toggle("active", button.dataset.page === page || (page === "document" && button.dataset.page === "knowledge"));
  });
  $("#currentCrumb").textContent = pageLabels[page] || "Ask";
  setSidebar(false);
  $("#contextMenu").hidden = true;
  if (page !== "ask") setInspector(false);
  if (page === "knowledge") renderKnowledge().catch((err) => notify(err.message));
  if (page === "document") renderDocument().catch((err) => notify(err.message));
  if (page === "pipeline") renderPipeline().catch((err) => notify(err.message));
  if (page === "analytics") renderAnalytics().catch((err) => notify(err.message));
  if (page === "settings") renderSettings().catch((err) => notify(err.message));
}

async function ensureSettings() {
  if (!state.settings) state.settings = await api.settings();
  return state.settings;
}

async function refreshShell() {
  const [settings, documents, conversations, index] = await Promise.all([
    api.settings(),
    api.documents(),
    api.conversations($("#threadSearch").value || ""),
    api.index(),
  ]);
  state.settings = settings;
  state.documents = documents.documents || [];
  state.conversations = conversations.conversations || [];
  state.index = index;
  if (!state.draftDirty) state.form = { ...settings };
  const scopeStillThere = state.documents.some(
    (doc) => doc.id === state.scopeId && doc.status === "indexed" && doc.included !== false
  );
  if (state.scopeId && !scopeStillThere) {
    state.scopeId = null;
    $("#contextLabel").textContent = "Add context";
  }
  $("#workspaceName").textContent = settings.workspace_name;
  $("#avatar").textContent = initials(settings.profile_name);
  $("#avatar").setAttribute("aria-label", `Profile for ${settings.profile_name}`);
  const version = String(index.version_id || "index").slice(0, 8);
  $("#syncLabel").textContent = `INDEX LIVE · ${version}`;
  renderThreads();
  renderCorpus();
}

function renderThreads() {
  const list = $("#threadList");
  const items = uniqueById(state.conversations);
  $("#threadCount").textContent = `${items.length} thread${items.length === 1 ? "" : "s"}`;
  if (!items.length) {
    list.innerHTML = `<p class="empty-note">No conversations yet. Ask a question to start one.</p>`;
    return;
  }
  list.innerHTML = items
    .map(
      (thread) => `
      <button type="button" class="thread ${thread.id === state.conversationId ? "active" : ""}" data-id="${thread.id}">
        <strong>${escapeHtml(thread.title)}</strong>
        <small>${escapeHtml(when(thread.updated_at).toUpperCase())}</small>
      </button>`
    )
    .join("");
  list.querySelectorAll(".thread").forEach((button) => {
    listen(button, "click", () => openConversation(button.dataset.id));
  });
}

function renderCorpus() {
  const indexed = state.documents.filter((doc) => doc.status === "indexed").slice(0, 4);
  $("#corpusList").innerHTML = indexed.length
    ? indexed
        .map(
          (doc) => `
        <button type="button" class="doc-stat" data-doc="${doc.id}" style="width:100%;background:transparent;text-align:left">
          <div class="doc-icon">${fileKind(doc.name)}</div>
          <div class="doc-meta"><strong>${escapeHtml(doc.name)}</strong><span>${doc.chunk_count} CHUNKS · ${escapeHtml(String(doc.status || "").toUpperCase())}</span></div>
          <i class="doc-ok"></i>
        </button>`
        )
        .join("")
    : `<p class="empty-note">No documents indexed yet.</p>`;
  $("#corpusList").querySelectorAll("[data-doc]").forEach((button) => {
    listen(button, "click", () => openDocument(button.dataset.doc));
  });
}

function orderedMessages(messages) {
  const seen = new Set();
  const result = [];
  for (const message of messages || []) {
    if (message?.id) {
      if (seen.has(message.id)) continue;
      seen.add(message.id);
    }
    result.push(message);
  }
  return result;
}

function conversationTurns(messages) {
  const turns = [];
  for (const message of orderedMessages(messages)) {
    if (message.role === "user") turns.push({ question: message, answer: null });
    else if (message.role === "assistant") {
      const open = turns[turns.length - 1];
      if (open && !open.answer) open.answer = message;
      else turns.push({ question: null, answer: message });
    }
  }
  return turns;
}

function renderAnswerHtml(text) {
  const safe = escapeHtml(text).replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>");
  return `<p>${safe.replace(/\[(\d+)\]/g, (_, n) => `<button type="button" class="citation" data-source="${n}" aria-label="View source ${n}">${n}</button>`)}</p>`;
}

function renderConversation() {
  const root = $("#conversationInner");
  const turns = conversationTurns(state.messages);
  if (!turns.length) {
    root.innerHTML = `
      <div class="home-empty">
        <div class="answer-kicker"><span>Grounded answers</span></div>
        <h2>Ask across the documents you have indexed.</h2>
        <p>Upload a source, then ask a question. Answers stay attached to the passages Jev keeps.</p>
      </div>`;
    $("#inspectorBody").innerHTML = `<p class="empty-note">Ask a question to see the evidence for the answer.</p>`;
    return;
  }
  let index = state.turnIndex;
  if (index == null || index < 0 || index >= turns.length) index = turns.length - 1;
  const turn = turns[index];
  const assistant = turn.answer;
  const question = turn.question;
  const earlier = turns
    .map((item, itemIndex) => ({ item, itemIndex }))
    .filter(({ itemIndex }) => itemIndex !== index);
  const validation = assistant?.validation || {};
  const sources = assistant?.sources || [];
  const history = earlier.length
    ? `<div class="turn-history">${earlier
        .map(
          ({ item, itemIndex }) =>
            `<button type="button" data-turn="${itemIndex}">${escapeHtml(item.question?.content || item.answer?.content || "Earlier turn")}</button>`
        )
        .join("")}</div>`
    : "";
  const body = assistant
    ? renderAnswerHtml(assistant.content || "")
    : `<p class="empty-note">No answer was stored for this question.</p>`;
  root.innerHTML = `
    ${history}
    <div class="answer-kicker"><span>${assistant ? (validation.passed ? "Grounded answer" : "Answer") : "Question"}</span></div>
    <h2 class="query-title">${escapeHtml(question?.content || "Question")}</h2>
    <div class="meta-line">
      <span class="meta-chip ${validation.passed ? "good" : ""}">${assistant ? (validation.passed ? "GROUNDED" : "CHECK FAILED") : "WAITING"}</span>
      <span class="meta-chip">${sources.length} SOURCE${sources.length === 1 ? "" : "S"}</span>
    </div>
    <article class="answer" id="answer">${body}</article>
    ${
      assistant
        ? `<div class="answer-actions">
      <div class="action-group">
        <button class="mini-action" id="copyAnswer" type="button" aria-label="Copy answer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
        <button class="mini-action ${assistant.rating === "helpful" ? "active" : ""}" id="helpful" type="button" aria-pressed="${assistant.rating === "helpful"}" aria-label="Helpful answer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M7 10v12H3V10h4zM7 20h10a2 2 0 0 0 2-1.6l1.4-7A2 2 0 0 0 18.4 9H14l1-4c.4-1.8-2-2.8-3-1.2L7 10z"/></svg></button>
        <button class="mini-action ${assistant.rating === "unhelpful" ? "active" : ""}" id="unhelpful" type="button" aria-pressed="${assistant.rating === "unhelpful"}" aria-label="Unhelpful answer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M17 14V2h4v12h-4zM17 4H7a2 2 0 0 0-2 1.6l-1.4 7A2 2 0 0 0 5.6 15H10l-1 4c-.4 1.8 2 2.8 3 1.2l5-6.2z"/></svg></button>
      </div>
      <span class="verified">${validation.passed ? "validation passed" : escapeHtml(validation.reason || "withheld")}</span>
    </div>`
        : ""
    }`;
  root.querySelectorAll("[data-turn]").forEach((button) => {
    listen(button, "click", () => {
      state.turnIndex = Number(button.dataset.turn);
      renderConversation();
    });
  });
  if (!assistant) {
    renderInspector(null);
    return;
  }
  root.querySelectorAll(".citation").forEach((button) => {
    listen(button, "click", () => openSource(button.dataset.source, assistant, button));
  });
  listen($("#copyAnswer"), "click", () => copyAnswer(assistant.content || ""));
  listen($("#helpful"), "click", () => sendFeedback(assistant, "helpful"));
  listen($("#unhelpful"), "click", () => sendFeedback(assistant, "unhelpful"));
  renderInspector(assistant);
}

function renderInspector(message) {
  const sources = message?.sources || [];
  const trace = message?.trace || {};
  const body = $("#inspectorBody");
  const showTrace = state.settings?.show_traces !== false;
  if (!message) {
    body.innerHTML = `<p class="empty-note">Ask a question to see the evidence for the answer.</p>`;
    return;
  }
  if (!showTrace && !sources.length) {
    body.innerHTML = `<p class="empty-note">Retrieval traces are hidden in settings.</p>`;
    return;
  }
  const traceBlock = showTrace
    ? `<section class="trace">
      <div class="trace-head"><span>Retrieval trace</span><span class="trace-time">${trace.candidates ?? 0} candidates</span></div>
      <div class="trace-stats">
        <div><strong>${trace.candidates ?? 0} → ${trace.kept ?? sources.length}</strong><span>chunks retained</span></div>
        <div><strong>${Number(trace.similarity_threshold ?? state.settings?.similarity_threshold ?? 0).toFixed(2)}</strong><span>min similarity</span></div>
        <div><strong>${escapeHtml(String(state.index?.version_id || "").slice(0, 8) || "—")}</strong><span>index version</span></div>
      </div>
    </section>`
    : "";
  body.innerHTML = `
    ${traceBlock}
    <div class="source-title"><strong>Supporting sources</strong><span>${sources.length} MATCHES</span></div>
    <div class="source-list">
      ${
        sources.length
          ? sources
              .map((source, index) => {
                const score = source.similarity_score != null ? `${Math.round(source.similarity_score * 100)}%` : "—";
                return `<button type="button" class="source" data-id="${index + 1}" data-doc="${escapeHtml(source.document_id || "")}">
                  <div class="source-top"><span class="source-type"><span class="file-badge">${fileKind(source.document_name)}</span>${escapeHtml(source.document_name || "Source")}</span><span class="score">${score}</span></div>
                  <h3>${escapeHtml(source.header_context || `Page ${source.page_number}`)}</h3>
                  <p>${escapeHtml((source.matched_passage || source.text || "").slice(0, 220))}</p>
                  <div class="source-foot"><span>PAGE ${source.page_number || "—"}</span><span>VIEW ↗</span></div>
                </button>`;
              })
              .join("")
          : `<p class="empty-note">No citation was kept for this answer.</p>`
      }
    </div>`;
  body.querySelectorAll(".source").forEach((button) => {
    listen(button, "click", () => {
      body.querySelectorAll(".source").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const mark = document.querySelector(`#answer .citation[data-source="${button.dataset.id}"]`);
      document.querySelectorAll("#answer .citation").forEach((item) => item.classList.remove("active"));
      mark?.classList.add("active");
      if (!button.dataset.doc) {
        notify("That source is not linked to a document.");
        return;
      }
      openDocument(button.dataset.doc);
    });
  });
}

function openSource(number, message, citation) {
  setInspector(true);
  renderInspector(message);
  document.querySelectorAll("#answer .citation").forEach((item) => item.classList.remove("active"));
  citation?.classList.add("active");
  const source = document.querySelector(`#inspectorBody .source[data-id="${number}"]`);
  if (!source) {
    notify(`No source is numbered ${number}.`);
    return;
  }
  source.classList.add("active");
  source.scrollIntoView({ block: "center" });
}

async function copyAnswer(text) {
  if (!text.trim()) throw new Error("There is no answer to copy.");
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard is unavailable in this browser.");
    await navigator.clipboard.writeText(text);
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-999px";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    if (!ok) throw new Error(err?.message || "Clipboard is unavailable in this browser.");
  }
  notify("Answer copied");
}

async function openConversation(id) {
  const data = await api.conversation(id);
  state.conversationId = id;
  state.messages = data.messages || [];
  state.turnIndex = null;
  showPage("ask");
  renderThreads();
  renderConversation();
}

async function sendFeedback(message, rating) {
  if (!message?.id) {
    notify("Ask again after the server has saved this answer.");
    return;
  }
  await api.feedback(message.id, rating);
  message.rating = rating;
  renderConversation();
  notify("Feedback recorded");
}

function consumeSse(block, handle) {
  for (const rawLine of block.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (!line.startsWith("data:")) continue;
    const payload = line.slice(5).trim();
    if (!payload || payload === "[DONE]") continue;
    let event;
    try {
      event = JSON.parse(payload);
    } catch {
      throw new Error("The answer stream was interrupted.");
    }
    handle(event);
  }
}

async function readChatStream(response, onStatus) {
  const reader = response.body?.getReader();
  if (!reader) throw new Error("The answer stream was interrupted.");
  const decoder = new TextDecoder();
  let buffer = "";
  const result = { answer: "", sources: [], validation: null, trace: null, messageId: null, error: null };
  const handle = (event) => {
    if (event.type === "conversation") state.conversationId = event.id;
    if (event.type === "status") onStatus(event.message || "Searching the index…");
    if (event.type === "sources") result.sources = event.data || [];
    if (event.type === "validation") result.validation = event;
    if (event.type === "trace") result.trace = event;
    if (event.type === "chunk") result.answer += event.content || "";
    if (event.type === "saved") result.messageId = event.message_id || null;
    if (event.type === "error") result.error = event.content || "Answer failed";
  };
  while (true) {
    const { done, value } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: !done });
    let splitAt = buffer.indexOf("\n\n");
    while (splitAt !== -1) {
      consumeSse(buffer.slice(0, splitAt), handle);
      buffer = buffer.slice(splitAt + 2);
      splitAt = buffer.indexOf("\n\n");
    }
    if (done) break;
  }
  buffer += decoder.decode();
  if (buffer.trim()) consumeSse(buffer, handle);
  if (result.error) throw new Error(result.error);
  return result;
}

async function reloadConversation() {
  if (!state.conversationId) return;
  const data = await api.conversation(state.conversationId);
  state.messages = data.messages || [];
  state.turnIndex = null;
}

async function sendQuestion(query) {
  if (state.busy) {
    notify("Still answering the previous question.");
    return;
  }
  state.busy = true;
  $("#askForm").querySelector(".send").disabled = true;
  $("#contextMenu").hidden = true;
  const root = $("#conversationInner");
  root.innerHTML = `<p class="status-line">Searching the index…</p><h2 class="query-title">${escapeHtml(query)}</h2>`;
  const prior = state.messages.slice();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        document_id: state.scopeId || null,
        conversation_id: state.conversationId,
      }),
    });
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(errorMessage(err, "Chat failed"));
    }
    const result = await readChatStream(response, (message) => {
      const line = root.querySelector(".status-line");
      if (line) line.textContent = message;
    });
    try {
      await reloadConversation();
    } catch (err) {
      state.messages = prior.concat([
        { role: "user", content: query },
        {
          id: result.messageId,
          role: "assistant",
          content: result.answer,
          sources: result.sources,
          validation: result.validation,
          trace: result.trace,
        },
      ]);
      notify(err.message);
    }
    await refreshShell();
    renderConversation();
  } catch (err) {
    let message = err.message;
    if (state.conversationId) {
      try {
        await reloadConversation();
      } catch (loadErr) {
        state.messages = prior;
        message = `${err.message} ${loadErr.message}`;
      }
    } else {
      state.messages = prior;
    }
    notify(message);
    renderConversation();
  } finally {
    state.busy = false;
    $("#askForm").querySelector(".send").disabled = false;
  }
}

async function renderKnowledge() {
  await refreshShell();
  const indexed = state.documents.filter((doc) => doc.status === "indexed");
  const processing = state.documents.filter((doc) => doc.status !== "indexed");
  const chunks = indexed.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
  const storage = indexed.reduce((sum, doc) => sum + (doc.bytes || 0), 0);
  $("#page-knowledge").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Knowledge operations</div><h1>Knowledge base</h1><p>Curate the source material that can be searched, cited, and used for grounded answers.</p></div>
        <div class="page-actions">
          <button class="btn" id="connectSource" type="button">Connect source</button>
          <button class="btn primary" id="uploadButton" type="button">Upload files</button>
        </div>
      </header>
      <div class="stat-grid">
        <article class="stat-card accent"><span>Total documents</span><div class="stat-value">${indexed.length}</div><small>${processing.length} still processing</small></article>
        <article class="stat-card"><span>Indexed chunks</span><div class="stat-value">${chunks}</div><small>Active index</small></article>
        <article class="stat-card"><span>Storage used</span><div class="stat-value">${bytes(storage)}</div><small>Original files kept locally</small></article>
        <article class="stat-card dark"><span>Index status</span><div class="stat-value">${state.index?.compatible ? "Healthy" : "Check"}</div><small>${escapeHtml(state.index?.embedding_model || "")}</small></article>
      </div>
      <section class="panel">
        <div class="panel-head">
          <div><h2>Documents</h2></div>
          <div class="panel-tools">
            <label class="filter-field"><input id="documentSearch" value="${escapeHtml(state.docQuery)}" placeholder="Search documents" aria-label="Search documents" /></label>
            <div class="segmented" id="docFilter">
              <button type="button" class="${state.docFilter === "all" ? "active" : ""}" data-filter="all">All</button>
              <button type="button" class="${state.docFilter === "indexed" ? "active" : ""}" data-filter="indexed">Synced</button>
              <button type="button" class="${state.docFilter === "processing" ? "active" : ""}" data-filter="processing">Processing</button>
            </div>
          </div>
        </div>
        <div class="table-scroll">
          <table class="data-table"><thead><tr><th>Document</th><th>Collection</th><th>Chunks</th><th>Status</th><th></th></tr></thead>
          <tbody id="docRows"></tbody></table>
        </div>
      </section>
    </div>`;
  listen($("#uploadButton"), "click", () => chooseFile("upload"));
  listen($("#connectSource"), "click", () => showConnect());
  listen($("#documentSearch"), "input", (event) => {
    state.docQuery = event.target.value;
    paintDocuments();
  });
  $("#docFilter").querySelectorAll("button").forEach((button) => {
    listen(button, "click", () => {
      state.docFilter = button.dataset.filter;
      $("#docFilter").querySelectorAll("button").forEach((item) => item.classList.toggle("active", item === button));
      paintDocuments();
    });
  });
  paintDocuments();
}

function filteredDocuments() {
  const q = state.docQuery.trim().toLowerCase();
  return state.documents.filter((doc) => {
    const name = String(doc.name || "").toLowerCase();
    const statusOk =
      state.docFilter === "all" ||
      (state.docFilter === "indexed" ? doc.status === "indexed" : doc.status !== "indexed");
    return statusOk && name.includes(q);
  });
}

function paintDocuments() {
  const body = $("#docRows");
  if (!body) return;
  const documents = filteredDocuments();
  if (!documents.length) {
    const note = state.documents.length
      ? "No documents match this filter."
      : "No documents yet. Upload a file to index it.";
    body.innerHTML = `<tr><td colspan="5"><p class="empty-note">${note}</p></td></tr>`;
    return;
  }
  body.innerHTML = documents
    .map(
      (doc) => `<tr class="document-row" data-id="${escapeHtml(doc.id)}" data-status="${escapeHtml(doc.status || "")}">
        <td><div class="file-cell"><div class="doc-icon">${fileKind(doc.name)}</div><div><strong>${escapeHtml(doc.name)}</strong><span>${bytes(doc.bytes)} · ${doc.max_page || 0} pages</span></div></div></td>
        <td>${escapeHtml(doc.collection || "General")}</td>
        <td>${doc.chunk_count || "—"}</td>
        <td><span class="status ${doc.status === "indexed" ? "" : "sync"}">${escapeHtml(doc.status || "unknown")}</span></td>
        <td><button class="row-action" type="button" aria-label="Open ${escapeHtml(doc.name)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 18 6-6-6-6"/></svg></button></td>
      </tr>`
    )
    .join("");
  body.querySelectorAll(".document-row").forEach((row) => {
    listen(row, "click", () => openListedDocument(row.dataset.id));
  });
}

function openListedDocument(id) {
  const doc = state.documents.find((item) => item.id === id);
  if (!doc) {
    notify("Document not found.");
    return;
  }
  if (doc.status !== "indexed") {
    notify(pendingCopy[doc.status] || "This document is not ready to open.");
    return;
  }
  openDocument(id);
}

async function openDocument(id) {
  state.activeDocumentId = id;
  showPage("document");
}

async function renderDocument() {
  const page = $("#page-document");
  if (!state.activeDocumentId) {
    page.innerHTML = `<div class="page-content"><p class="empty-note">Choose a document from the knowledge base.</p></div>`;
    return;
  }
  let doc;
  try {
    doc = await api.document(state.activeDocumentId);
  } catch (err) {
    page.innerHTML = `<div class="page-content"><button class="btn small" id="backKnowledge" type="button">← Back to knowledge base</button><p class="empty-note">${escapeHtml(err.message)}</p></div>`;
    listen($("#backKnowledge"), "click", () => showPage("knowledge"));
    throw err;
  }
  const passages = (doc.passages || []).slice(0, 8);
  page.innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div>
          <button class="btn small" id="backKnowledge" type="button">← Back to knowledge base</button>
          <h1 style="font-size:46px">${escapeHtml(doc.name)}</h1>
          <p>${escapeHtml(doc.collection || "General")} · ${doc.chunk_count || 0} chunks · ${doc.max_page || 0} pages</p>
        </div>
        <div class="page-actions">
          <button class="btn" id="replaceFile" type="button">Replace file</button>
          <button class="btn dark" id="openOriginal" type="button">Open original</button>
          <button class="btn danger" id="deleteDoc" type="button">Delete</button>
        </div>
      </header>
      <div class="doc-layout">
        <article class="panel document-preview">${escapeHtml(doc.preview || "No extractable preview.").replace(/\n/g, "<br>")}</article>
        <aside class="stack">
          <section class="panel">
            <div class="panel-head"><div><h2>Index record</h2><p>Saved with this document</p></div></div>
            <div class="form-block">
              <div class="switch-row"><div><strong>Included in answers</strong><p>When off, retrieval skips this file</p></div><button type="button" class="switch ${doc.included ? "on" : ""}" id="includedSwitch" role="switch" aria-pressed="${doc.included ? "true" : "false"}"></button></div>
              <div class="switch-row"><div><strong>Citation required</strong><p>Stored with the document record</p></div><button type="button" class="switch ${doc.citation_required ? "on" : ""}" id="citeSwitch" role="switch" aria-pressed="${doc.citation_required ? "true" : "false"}"></button></div>
            </div>
          </section>
          <section class="panel">
            <div class="panel-head"><div><h2>Matched chunks</h2><p>${(doc.passages || []).length} parents loaded</p></div></div>
            <div class="chunk-list">${passages.map((item, index) => `<div class="chunk ${index === 0 ? "active" : ""}"><span>PAGE ${item.page_number || "—"} · ${escapeHtml(item.header_context || "Passage")}</span><p>${escapeHtml(String(item.text || "").slice(0, 280))}</p></div>`).join("") || `<p class="empty-note">No chunks stored.</p>`}</div>
          </section>
        </aside>
      </div>
    </div>`;
  listen($("#backKnowledge"), "click", () => showPage("knowledge"));
  listen($("#replaceFile"), "click", () => chooseFile("replace", doc.id));
  listen($("#openOriginal"), "click", () => openOriginal(doc));
  listen($("#deleteDoc"), "click", () => confirmDelete(doc));
  listen($("#includedSwitch"), "click", (event) =>
    togglePolicy(event.currentTarget, doc.id, "included", "Document included in answers", "Document excluded from answers")
  );
  listen($("#citeSwitch"), "click", (event) =>
    togglePolicy(event.currentTarget, doc.id, "citation_required", "Citations are required for this document", "Citations are optional for this document")
  );
}

async function togglePolicy(button, id, key, onText, offText) {
  const next = !button.classList.contains("on");
  button.disabled = true;
  try {
    await api.updateDocument(id, { [key]: next });
    button.classList.toggle("on", next);
    button.setAttribute("aria-pressed", next ? "true" : "false");
    notify(next ? onText : offText);
    await refreshShell();
  } catch (err) {
    notify(err.message);
  } finally {
    button.disabled = false;
  }
}

async function openOriginal(doc) {
  const response = await fetch(`/api/documents/${doc.id}/file`);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(errorMessage(data, "The original file is not stored for this document."));
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const header = response.headers.get("Content-Disposition") || "";
  const named = /filename="?([^";]+)"?/i.exec(header);
  const opened = window.open(url, "_blank", "noopener");
  if (!opened) {
    const link = document.createElement("a");
    link.href = url;
    link.download = named?.[1] || doc.name || "document";
    document.body.appendChild(link);
    link.click();
    link.remove();
  }
  setTimeout(() => URL.revokeObjectURL(url), 60000);
}

function pipeRow(nodes) {
  return `<div class="pipeline-row">${nodes.map((node) => node).join(pipeArrow)}</div>`;
}

async function renderPipeline() {
  const [index, settings] = await Promise.all([api.index(), ensureSettings()]);
  state.index = index;
  state.settings = settings;
  $("#page-pipeline").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Index architecture</div><h1>Pipeline</h1><p>Ingestion, retrieval, Jev reranking, and the grounding check for the active index.</p></div>
        <div class="page-actions"><button class="btn primary" id="refreshIndex" type="button">Refresh index</button></div>
      </header>
      <div class="stat-grid">
        <article class="stat-card"><span>Active version</span><div class="stat-value">${escapeHtml(String(index.version_id || "").slice(0, 8) || "—")}</div><small>${escapeHtml(index.embedding_model || "")} · ${index.embedding_dimensions ?? "—"} dims</small></article>
        <article class="stat-card accent"><span>Jev</span><div class="stat-value">${index.jev_configured ? "Ready" : "Key"}</div><small>${escapeHtml(index.jev_model || "")}</small></article>
        <article class="stat-card"><span>Documents</span><div class="stat-value">${index.documents ?? 0}</div><small>${index.chunks ?? 0} chunks</small></article>
        <article class="stat-card dark"><span>Compatibility</span><div class="stat-value">${index.compatible ? "Match" : "Blocked"}</div><small>Similarity floor ${settings.similarity_threshold}</small></article>
      </div>
      <div class="two-column">
        <section class="panel">
          <div class="panel-head"><div><h2>Live path</h2><p>What a question actually runs</p></div><span class="status">${index.compatible ? "Ready" : "Needs refresh"}</span></div>
          <div class="pipeline-map">
            <div class="branch-label">Ingestion</div>
            ${pipeRow([
              `<div class="pipe-node on"><span class="node-label">Source</span><strong>Documents</strong><small>${index.documents ?? 0} active</small></div>`,
              `<div class="pipe-node on"><span class="node-label">Prepare</span><strong>${escapeHtml(settings.chunking || "Parent-child")}</strong><small>${index.chunks ?? 0} chunks</small></div>`,
              `<div class="pipe-node on"><span class="node-label">Encode</span><strong>Tokenize + embed</strong><small>${index.embedding_dimensions ?? "—"} dims</small></div>`,
              `<div class="pipe-node on"><span class="node-label">Store</span><strong>Vector index</strong><small>${escapeHtml(String(index.version_id || "").slice(0, 8) || "—")}</small></div>`,
            ])}
            <div class="branch-label" style="margin-top:28px">Query</div>
            ${pipeRow([
              `<div class="pipe-node"><span class="node-label">Retrieve</span><strong>Top-k search</strong><small>k = ${settings.top_k}</small></div>`,
              `<div class="pipe-node"><span class="node-label">Refine</span><strong>Jev reranker</strong><small>keep ${settings.max_parents}</small></div>`,
              `<div class="pipe-node"><span class="node-label">Answer</span><strong>${escapeHtml(index.answer_model || "OpenRouter")}</strong><small>grounding check</small></div>`,
            ])}
          </div>
        </section>
        <section class="panel">
          <div class="panel-head"><div><h2>Recent runs</h2><p>Uploads, deletes, and handoffs</p></div></div>
          <div class="run-list">${(index.runs || []).map((run) => `<div class="run"><span class="run-time">${escapeHtml(when(run.created_at))}</span><div><strong>${escapeHtml(run.name)}</strong><p>${escapeHtml(run.detail)}</p></div><span class="status">${escapeHtml(run.status)}</span></div>`).join("") || `<p class="empty-note">No runs recorded yet.</p>`}</div>
        </section>
      </div>
    </div>`;
  listen($("#refreshIndex"), "click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Refreshing…";
    try {
      await api.refreshIndex();
      notify("Index refreshed");
      await renderPipeline();
      await refreshShell();
    } catch (err) {
      notify(err.message);
      button.disabled = false;
      button.textContent = "Refresh index";
    }
  });
}

async function renderAnalytics() {
  const report = await api.analytics(state.analyticsDays);
  const series = report.series || [];
  const max = Math.max(1, ...series.map((point) => Number(point.questions) || 0));
  const gaps = report.gaps || [];
  $("#page-analytics").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Quality intelligence</div><h1>Analytics</h1><p>Counts come from questions this workspace has actually asked.</p></div>
        <div class="page-actions">
          <div class="segmented" id="range">${[7, 30, 90].map((days) => `<button type="button" data-days="${days}" class="${days === state.analyticsDays ? "active" : ""}">${days}D</button>`).join("")}</div>
          <button class="btn" id="exportReport" type="button">Export report</button>
        </div>
      </header>
      <div class="stat-grid">
        <article class="stat-card dark"><span>Questions answered</span><div class="stat-value">${report.questions ?? 0}</div><small>in ${report.days || state.analyticsDays} days</small></article>
        <article class="stat-card accent"><span>Grounded answer rate</span><div class="stat-value">${percent(report.grounded_rate)}</div><small>${report.questions ?? 0} recorded</small></article>
        <article class="stat-card"><span>Helpful rating</span><div class="stat-value">${report.ratings ? percent(report.helpful_rate) : "—"}</div><small>${report.ratings ?? 0} ratings</small></article>
        <article class="stat-card"><span>Withheld rate</span><div class="stat-value">${percent(report.withheld_rate)}</div><small>p50 retrieval ${report.retrieval_p50_ms ?? 0} ms</small></article>
      </div>
      <section class="panel">
        <div class="panel-head"><div><h2>Daily questions</h2><p>Grounded answers are the lighter bar</p></div></div>
        <div class="chart">${series.map((point) => `<div class="bar-group" title="${escapeHtml(point.day)}"><i class="bar" style="height:${Math.round(((Number(point.questions) || 0) / max) * 100)}%"></i><i class="bar secondary" style="height:${Math.round(((Number(point.grounded) || 0) / max) * 100)}%"></i></div>`).join("") || `<p class="empty-note">Ask a few questions to fill this chart.</p>`}</div>
      </section>
      <section class="panel" style="margin-top:14px">
        <div class="panel-head"><div><h2>Knowledge gaps</h2><p>Questions that were not grounded</p></div><button class="btn small" id="gapUpload" type="button">Add source</button></div>
        <div class="table-scroll"><table class="data-table"><thead><tr><th>Question</th><th>Attempts</th><th>Best vector score</th></tr></thead><tbody>
          ${gaps.map((gap) => `<tr><td><strong>${escapeHtml(gap.query)}</strong></td><td>${gap.attempts}</td><td>${gap.best_similarity == null ? "—" : Number(gap.best_similarity).toFixed(2)}</td></tr>`).join("") || `<tr><td colspan="3"><p class="empty-note">No withheld questions in this range.</p></td></tr>`}
        </tbody></table></div>
      </section>
    </div>`;
  $("#range").querySelectorAll("button").forEach((button) => {
    listen(button, "click", () => {
      state.analyticsDays = Number(button.dataset.days);
      return renderAnalytics();
    });
  });
  listen($("#exportReport"), "click", () => exportReport());
  listen($("#gapUpload"), "click", () => chooseFile("upload"));
}

async function exportReport() {
  const response = await fetch(`/api/analytics/export?days=${state.analyticsDays}`);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(errorMessage(data, "Export failed."));
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `needle-analytics-${state.analyticsDays}d.csv`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const settingFields = [
  ["setName", "workspace_name"],
  ["setProfile", "profile_name"],
  ["setCollection", "default_collection"],
  ["setRegion", "region"],
  ["setLength", "answer_length"],
  ["setCiteStyle", "citation_style"],
  ["setTopK", "top_k"],
  ["setSim", "similarity_threshold"],
  ["setParents", "max_parents"],
  ["setChunk", "chunking"],
];

function applySettingsForm() {
  const page = $("#page-settings");
  const form = state.form;
  if (!page || !form) return;
  settingFields.forEach(([id, key]) => {
    const node = page.querySelector(`#${id}`);
    if (node && document.activeElement !== node) node.value = form[key] ?? "";
  });
  page.querySelectorAll("[data-switch]").forEach((button) => {
    const on = Boolean(form[button.dataset.switch]);
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
}

function syncSettingsControl(node, key) {
  if (!state.form) state.form = { ...(state.settings || {}) };
  state.form[key] = node.type === "number" ? Number(node.value) : node.value;
  state.draftDirty = true;
}

function settingsPayload() {
  if (!state.form || !$("#page-settings")?.childElementCount) {
    throw new Error("Open Settings before saving. The form is not on screen.");
  }
  const form = state.form;
  const current = state.settings || {};
  const number = (key, label, low, high) => {
    const value = Number(form[key]);
    if (!Number.isFinite(value) || value < low || value > high) {
      throw new Error(`${label} must be between ${low} and ${high}.`);
    }
    return value;
  };
  return {
    workspace_name: String(form.workspace_name ?? "").trim(),
    profile_name: String(form.profile_name ?? "").trim(),
    default_collection: String(form.default_collection ?? "").trim() || "General",
    region: form.region || current.region || "European Union",
    timezone: current.timezone || form.timezone || "UTC",
    show_traces: Boolean(form.show_traces),
    allow_downloads: Boolean(form.allow_downloads),
    answer_length: form.answer_length || current.answer_length || "Balanced",
    citation_style: form.citation_style || current.citation_style || "Inline numbered",
    require_citations: Boolean(form.require_citations),
    withhold_ungrounded: Boolean(form.withhold_ungrounded),
    top_k: number("top_k", "Top-k candidates", 1, 50),
    similarity_threshold: number("similarity_threshold", "Similarity threshold", 0, 1),
    max_parents: number("max_parents", "Reranked context limit", 1, 12),
    chunking: form.chunking || current.chunking || "Parent-child",
  };
}

async function renderSettings() {
  const [settings, members, integrations] = await Promise.all([api.settings(), api.members(), api.integrations()]);
  state.settings = settings;
  const previous = state.draftDirty ? state.form : null;
  state.form = { ...settings, ...(previous || {}) };
  const switches = (name) => `<button type="button" class="switch" data-switch="${name}" role="switch" aria-pressed="false"></button>`;
  $("#page-settings").innerHTML = `
    <div class="page-content">
      <header class="page-header"><div><div class="section-eyebrow">Workspace control</div><h1>Settings</h1><p>These values are stored locally and used by the next question.</p></div><div class="page-actions"><button class="btn primary" id="saveSettings" type="button">Save changes</button></div></header>
      <div class="settings-layout">
        <nav class="settings-nav" aria-label="Settings sections">
          <button type="button" class="active" data-settings="general">General</button>
          <button type="button" data-settings="answers">Answer behavior</button>
          <button type="button" data-settings="retrieval">Retrieval</button>
          <button type="button" data-settings="members">Members</button>
          <button type="button" data-settings="integrations">Integrations</button>
          <button type="button" data-settings="billing">Data</button>
        </nav>
        <div>
          <section class="panel settings-section active" data-section="general">
            <div class="panel-head"><div><h2>General</h2><p>Workspace identity</p></div></div>
            <div class="form-block"><div class="field-grid">
              <div class="field"><label for="setName">Workspace name</label><input id="setName" /></div>
              <div class="field"><label for="setProfile">Your name</label><input id="setProfile" /></div>
              <div class="field"><label for="setCollection">Default collection</label><select id="setCollection">${optionList(["General", "Product", "Security", "Research"], state.form.default_collection)}</select></div>
              <div class="field"><label for="setRegion">Region</label><select id="setRegion">${optionList(["European Union", "United States", "Asia Pacific"], state.form.region)}</select></div>
            </div>
            <div class="switch-row"><div><strong>Show retrieval traces</strong><p>Evidence panel includes candidate counts</p></div>${switches("show_traces")}</div>
            <div class="switch-row"><div><strong>Allow source downloads</strong><p>Open original files from document detail</p></div>${switches("allow_downloads")}</div>
            </div>
          </section>
          <section class="panel settings-section" data-section="answers">
            <div class="panel-head"><div><h2>Answer behavior</h2></div></div>
            <div class="form-block">
              <div class="field-grid">
                <div class="field"><label for="setLength">Default answer length</label><select id="setLength">${optionList(["Concise", "Balanced", "Detailed"], state.form.answer_length)}</select></div>
                <div class="field"><label for="setCiteStyle">Citation style</label><select id="setCiteStyle">${optionList(["Inline numbered", "Footnotes", "Source cards"], state.form.citation_style)}</select></div>
              </div>
              <div class="switch-row"><div><strong>Require citations</strong><p>Prompt the model to cite [1], [2]</p></div>${switches("require_citations")}</div>
              <div class="switch-row"><div><strong>No-answer fallback</strong><p>Withhold drafts that fail the grounding check</p></div>${switches("withhold_ungrounded")}</div>
            </div>
          </section>
          <section class="panel settings-section" data-section="retrieval">
            <div class="panel-head"><div><h2>Retrieval</h2><p>Used on the next question. Chunking applies to the next upload.</p></div></div>
            <div class="form-block"><div class="field-grid">
              <div class="field"><label for="setTopK">Top-k candidates</label><input id="setTopK" type="number" min="1" max="50" /></div>
              <div class="field"><label for="setSim">Similarity threshold</label><input id="setSim" type="number" min="0" max="1" step="0.01" /></div>
              <div class="field"><label for="setParents">Reranked context limit</label><input id="setParents" type="number" min="1" max="12" /></div>
              <div class="field"><label for="setChunk">Chunking strategy</label><select id="setChunk">${optionList(["Parent-child", "Fixed window", "Index card summary"], state.form.chunking)}</select></div>
            </div></div>
          </section>
          <section class="panel settings-section" data-section="members">
            <div class="panel-head"><div><h2>Members</h2><p>Local workspace directory</p></div><button class="btn small primary" id="inviteMember" type="button">Invite member</button></div>
            <div class="table-scroll"><table class="data-table"><thead><tr><th>Member</th><th>Role</th></tr></thead><tbody>
              ${(members.members || []).map((member) => `<tr><td><strong>${escapeHtml(member.name)}</strong><br><small>${escapeHtml(member.email)}</small></td><td>${escapeHtml(member.role)}</td></tr>`).join("") || `<tr><td colspan="2"><p class="empty-note">No members yet.</p></td></tr>`}
            </tbody></table></div>
          </section>
          <section class="panel settings-section" data-section="integrations">
            <div class="panel-head"><div><h2>Integrations</h2><p>Only file upload is connected on this server</p></div></div>
            <div class="form-block">${(integrations.integrations || [])
              .map(
                (item) => `<div class="switch-row"><div><strong>${escapeHtml(item.name)}</strong><p>${escapeHtml(item.detail)}</p></div><button class="btn small" type="button" data-integration="${escapeHtml(item.id)}" data-connected="${item.connected ? "true" : "false"}" data-detail="${escapeHtml(item.detail || "")}">${item.connected ? "Use" : "Connect"}</button></div>`
              )
              .join("")}</div>
          </section>
          <section class="panel settings-section" data-section="billing">
            <div class="panel-head"><div><h2>Local data</h2><p>This workspace runs on this machine</p></div></div>
            <div class="form-block"><h3>Danger zone</h3><p>Removes indexed documents, stored files, conversations, and saved settings.</p><button class="btn danger" id="resetWorkspace" type="button">Delete workspace data</button></div>
          </section>
        </div>
      </div>
    </div>`;
  applySettingsForm();
  const page = $("#page-settings");
  page.querySelectorAll(".settings-nav button").forEach((button) => {
    listen(button, "click", () => {
      page.querySelectorAll(".settings-nav button").forEach((item) => item.classList.remove("active"));
      page.querySelectorAll(".settings-section").forEach((section) => section.classList.remove("active"));
      button.classList.add("active");
      const section = page.querySelector(`[data-section="${button.dataset.settings}"]`);
      section?.classList.add("active");
      applySettingsForm();
    });
  });
  settingFields.forEach(([id, key]) => {
    const node = page.querySelector(`#${id}`);
    listen(node, "input", () => syncSettingsControl(node, key));
    listen(node, "change", () => syncSettingsControl(node, key));
  });
  page.querySelectorAll("[data-switch]").forEach((button) => {
    listen(button, "click", () => {
      button.classList.toggle("on");
      const on = button.classList.contains("on");
      button.setAttribute("aria-pressed", on ? "true" : "false");
      if (!state.form) state.form = { ...(state.settings || {}) };
      state.form[button.dataset.switch] = on;
      state.draftDirty = true;
    });
  });
  listen($("#saveSettings"), "click", () => saveSettings());
  listen($("#inviteMember"), "click", () => inviteDialog());
  listen($("#resetWorkspace"), "click", () => resetDialog());
  page.querySelectorAll("[data-integration]").forEach((button) => {
    listen(button, "click", () => {
      if (button.dataset.integration === "files" && button.dataset.connected === "true") {
        chooseFile("upload");
        return;
      }
      notify(button.dataset.detail || "Not configured on this server.");
    });
  });
}

async function saveSettings() {
  const payload = settingsPayload();
  const saved = await api.saveSettings(payload);
  state.settings = saved;
  state.form = { ...saved };
  state.draftDirty = false;
  applySettingsForm();
  notify("Workspace settings saved");
  await refreshShell();
}

function openModal(title, body, actions) {
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = body;
  const bar = $("#modalActions");
  bar.innerHTML = "";
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `btn ${action.primary ? "primary" : ""} ${action.danger ? "danger" : ""}`;
    button.textContent = action.label;
    button.addEventListener("click", () => {
      Promise.resolve(action.onClick()).catch((err) => notify(err.message || "Request failed"));
    });
    bar.appendChild(button);
  });
  $("#modal").hidden = false;
  $("#modalBody").querySelector("input")?.focus();
}

function closeModal() {
  $("#modal").hidden = true;
  $("#modalBody").innerHTML = "";
  $("#modalActions").innerHTML = "";
}

function showConnect() {
  openModal(
    "Connect a source",
    `<p>File upload is the connected source on this server. Google Drive, Notion, and Slack are not configured.</p>`,
    [
      { label: "Cancel", onClick: closeModal },
      {
        label: "Upload a file",
        primary: true,
        onClick: () => {
          closeModal();
          chooseFile("upload");
        },
      },
    ]
  );
}

function inviteDialog() {
  openModal(
    "Invite a member",
    `<div class="field" style="margin-bottom:8px"><label for="inviteName">Name</label><input id="inviteName" /></div><div class="field"><label for="inviteEmail">Email</label><input id="inviteEmail" type="email" /></div>`,
    [
      { label: "Cancel", onClick: closeModal },
      {
        label: "Invite",
        primary: true,
        onClick: async () => {
          await api.invite({ name: $("#inviteName").value.trim(), email: $("#inviteEmail").value.trim(), role: "Member" });
          closeModal();
          notify("Member added");
          await renderSettings();
        },
      },
    ]
  );
}

function confirmDelete(doc) {
  openModal(`Delete ${doc.name}?`, `<p>This removes the file, its vectors, and its keyword index.</p>`, [
    { label: "Cancel", onClick: closeModal },
    {
      label: "Delete",
      danger: true,
      onClick: async () => {
        await api.deleteDocument(doc.id);
        closeModal();
        state.activeDocumentId = null;
        if (state.scopeId === doc.id) {
          state.scopeId = null;
          $("#contextLabel").textContent = "Add context";
        }
        notify("Document deleted");
        showPage("knowledge");
      },
    },
  ]);
}

function resetDialog() {
  openModal("Delete workspace data?", `<p>This cannot be undone. Indexed files, conversations, and settings will be removed.</p>`, [
    { label: "Cancel", onClick: closeModal },
    {
      label: "Delete data",
      danger: true,
      onClick: async () => {
        await api.reset();
        closeModal();
        state.conversationId = null;
        state.messages = [];
        state.turnIndex = null;
        state.scopeId = null;
        state.form = null;
        state.draftDirty = false;
        state.activeDocumentId = null;
        $("#contextLabel").textContent = "Add context";
        $("#threadSearch").value = "";
        notify("Workspace data removed");
        await refreshShell();
        renderConversation();
        showPage("ask");
      },
    },
  ]);
}

function renderContextMenu() {
  const menu = $("#contextMenu");
  const docs = state.documents.filter((doc) => doc.status === "indexed" && doc.included !== false);
  menu.innerHTML = `<button type="button" data-scope="">All documents</button>${docs
    .map((doc) => `<button type="button" data-scope="${doc.id}">${escapeHtml(doc.name)}</button>`)
    .join("")}`;
  const selected = state.scopeId || "";
  menu.querySelectorAll("button").forEach((button) => {
    button.classList.toggle("active", (button.dataset.scope || "") === selected);
    listen(button, "click", () => {
      state.scopeId = button.dataset.scope || null;
      $("#contextLabel").textContent = state.scopeId ? "1 document" : "Add context";
      menu.hidden = true;
    });
  });
}

function chooseFile(mode, documentId = null) {
  if (state.uploading) {
    notify("A file is already being indexed.");
    return;
  }
  state.replaceId = mode === "replace" ? documentId : null;
  const input = $("#fileInput");
  input.value = "";
  input.click();
}

let recognition = null;
let threadSearchSeq = 0;

listen($("#askForm"), "submit", (event) => {
  event.preventDefault();
  const value = $("#askInput").value.trim();
  if (!value) {
    $("#askInput").focus();
    return;
  }
  $("#askInput").value = "";
  $("#askInput").style.height = "auto";
  return sendQuestion(value);
});

listen($("#askInput"), "input", () => {
  const input = $("#askInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 120)}px`;
});

listen($("#askInput"), "keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    $("#askForm").requestSubmit();
  }
});

listen($("#newChat"), "click", async () => {
  const created = await api.createConversation();
  state.conversationId = created.id;
  state.messages = [];
  state.turnIndex = null;
  $("#threadSearch").value = "";
  await refreshShell();
  showPage("ask");
  renderConversation();
  $("#askInput").focus();
});

listen($("#openKnowledge"), "click", () => showPage("knowledge"));
listen($("#homeMark"), "click", () => showPage("ask"));
document.querySelectorAll(".rail [data-page]").forEach((button) => listen(button, "click", () => showPage(button.dataset.page)));
listen($("#sourceToggle"), "click", () => setInspector(!$("#inspector").classList.contains("open")));
listen($("#closeInspector"), "click", () => setInspector(false));
listen($("#threadsToggle"), "click", () => setSidebar(!$("#sidebar").classList.contains("open")));
  listen($("#threadSearch"), "input", async (event) => {
  const seq = ++threadSearchSeq;
  try {
    const data = await api.conversations(event.target.value);
    if (seq !== threadSearchSeq) return;
    state.conversations = data.conversations || [];
    renderThreads();
  } catch (err) {
    if (seq === threadSearchSeq) throw err;
  }
});
listen($("#addContext"), "click", () => {
  const menu = $("#contextMenu");
  if (menu.hidden) renderContextMenu();
  menu.hidden = !menu.hidden;
});
listen($("#voiceButton"), "click", () => {
  const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Speech) {
    notify("Voice input is not available in this browser.");
    return;
  }
  if (recognition) {
    recognition.stop();
    return;
  }
  const session = new Speech();
  recognition = session;
  session.lang = navigator.language || "en-US";
  session.onresult = (event) => {
    const text = event.results?.[0]?.[0]?.transcript || "";
    if (!text) {
      notify("No speech was heard.");
      return;
    }
    const input = $("#askInput");
    input.value = text;
    input.dispatchEvent(new Event("input"));
  };
  session.onerror = (event) => {
    const code = event.error || "";
    const messages = {
      "not-allowed": "Microphone access was blocked.",
      "service-not-allowed": "Voice input is not available in this browser.",
      network: "Voice input could not reach the speech service.",
      "no-speech": "No speech was heard.",
    };
    if (code && code !== "aborted") notify(messages[code] || "Voice input stopped.");
  };
  session.onend = () => {
    if (recognition === session) recognition = null;
    $("#voiceButton").setAttribute("aria-pressed", "false");
  };
  try {
    session.start();
    $("#voiceButton").setAttribute("aria-pressed", "true");
    notify("Listening…");
  } catch (err) {
    recognition = null;
    $("#voiceButton").setAttribute("aria-pressed", "false");
    notify(err.message || "Voice input is not available in this browser.");
  }
});
listen($("#fileInput"), "change", async (event) => {
  const file = event.target.files?.[0];
  const replacing = state.replaceId;
  state.replaceId = null;
  event.target.value = "";
  if (!file) return;
  if (state.uploading) {
    notify("A file is already being indexed.");
    return;
  }
  state.uploading = true;
  notify(`Indexing ${file.name}…`);
  try {
    const uploaded = await api.upload(file);
    let message = `${uploaded.name} indexed`;
    if (replacing && replacing !== uploaded.id) {
      try {
        await api.deleteDocument(replacing);
      } catch (err) {
        message = `${uploaded.name} indexed, but the previous document was not removed. ${err.message}`;
      }
    }
    notify(message);
    await refreshShell();
    state.activeDocumentId = uploaded.id;
    showPage("document");
  } catch (err) {
    notify(err.message);
  } finally {
    state.uploading = false;
  }
});
listen($("#modal"), "click", (event) => {
  if (event.target.id === "modal") closeModal();
});
document.addEventListener("click", (event) => {
  const sidebar = $("#sidebar");
  const toggle = $("#threadsToggle");
  if (sidebar.classList.contains("open") && !sidebar.contains(event.target) && !toggle.contains(event.target)) {
    setSidebar(false);
  }
  const menu = $("#contextMenu");
  if (!menu.hidden && !$("#addContext").contains(event.target) && !menu.contains(event.target)) menu.hidden = true;
});
document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    showPage("ask");
    if (window.matchMedia("(max-width: 820px)").matches) setSidebar(true);
    $("#threadSearch").focus();
  }
  if (event.key === "Escape") {
    setInspector(false);
    setSidebar(false);
    $("#contextMenu").hidden = true;
    closeModal();
  }
});

$("#threadsToggle").setAttribute("aria-expanded", "false");
$("#sourceToggle").setAttribute("aria-expanded", "false");
$("#voiceButton").setAttribute("aria-pressed", "false");

refreshShell()
  .then(() => renderConversation())
  .catch((err) => notify(err.message));
