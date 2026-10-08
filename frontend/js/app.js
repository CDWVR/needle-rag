import { api, ApiError, errorMessage } from "./api.js";

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

const percent = (value) => (value == null ? "—" : `${Math.round(Number(value) * 1000) / 10}%`);

const seconds = (ms) => (ms == null ? "" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)}s`);

const isToday = (iso) => {
  const date = new Date(iso || "");
  return !Number.isNaN(date.getTime()) && date.toDateString() === new Date().toDateString();
};

const clock = (iso) => {
  const date = new Date(iso || "");
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
};

const shortDay = (iso) => {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString([], { month: "short", day: "2-digit" }).toUpperCase();
};

// Change against the previous period, phrased the way the stat cards read: "↑ 3.2 pts from prior period".
const delta = (current, prior, { points = true, invert = false } = {}) => {
  if (current == null || prior == null) return "";
  const diff = (Number(current) - Number(prior)) * (points ? 100 : 1);
  if (Math.abs(diff) < 0.05) return "No change from prior period";
  const up = diff > 0;
  const good = invert ? !up : up;
  return `<span class="${good ? "" : "stat-bad"}">${up ? "↑" : "↓"} ${Math.abs(diff).toFixed(1)}${points ? " pts" : ""}</span> from prior period`;
};

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

function showPage(page, { updateHash = true } = {}) {
  if (!pageLabels[page]) page = "ask";
  state.page = page;
  if (updateHash) {
    const hash = page === "document" && state.activeDocumentId ? `#document/${state.activeDocumentId}` : `#${page}`;
    if (location.hash !== hash) history.pushState(null, "", hash);
  }
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
  document.querySelector(`#page-${page}`)?.scrollTo(0, 0);
}

function routeFromHash() {
  const [page, id] = location.hash.replace(/^#/, "").split("/");
  if (page === "document" && id) state.activeDocumentId = decodeURIComponent(id);
  showPage(page || "ask", { updateHash: false });
}

window.addEventListener("popstate", routeFromHash);

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
  const row = (thread) => {
    const sourced = Number(thread.source_threads) || 0;
    const meta = [when(thread.updated_at).toUpperCase(), sourced ? `${sourced} SOURCED` : ""].filter(Boolean).join(" · ");
    return `
      <button type="button" class="thread ${thread.id === state.conversationId ? "active" : ""}" data-id="${escapeHtml(thread.id)}">
        <strong>${escapeHtml(thread.title)}</strong>
        <small>${escapeHtml(meta)}</small>
      </button>`;
  };
  const today = items.filter((thread) => isToday(thread.updated_at));
  const earlier = items.filter((thread) => !isToday(thread.updated_at));
  const group = (label, threads) =>
    threads.length
      ? `<div class="thread-label"><span class="section-eyebrow">${label}</span></div>${threads.map(row).join("")}`
      : "";
  list.innerHTML = group("Today", today) + group("Previous", earlier);
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
        <button type="button" class="doc-stat" data-doc="${escapeHtml(doc.id)}" style="width:100%;background:transparent;text-align:left">
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

function activeConversationTitle() {
  const thread = (state.conversations || []).find((item) => item.id === state.conversationId);
  const title = String(thread?.title || "").trim();
  return title || "Ask";
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
  const elapsed = assistant?.trace?.latencies_ms?.total;
  const callout = answerCallout(assistant, validation);
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
  const sessionTitle = activeConversationTitle();
  const asked = question?.content && question.content.trim() !== sessionTitle.trim()
    ? `<p class="user-prompt"><span class="section-eyebrow">You asked</span>${escapeHtml(question.content)}</p>`
    : "";
  root.innerHTML = `
    ${history}
    <div class="answer-kicker"><span>${assistant ? (validation.declined ? "Not in your sources" : validation.passed ? "Grounded answer" : "Answer") : "Question"}</span></div>
    <h2 class="query-title">${escapeHtml(sessionTitle)}</h2>
    ${asked}
    <div class="meta-line">
      <span class="meta-chip ${validation.passed && !validation.declined ? "good" : ""}">${assistant ? (validation.declined ? "NOT IN SOURCES" : validation.passed ? "GROUNDED" : validation.reject_category === "retrieval_abstain" ? "NO EVIDENCE" : "CHECK FAILED") : "WAITING"}</span>
      ${validation.confidence ? `<span class="meta-chip">${escapeHtml(String(validation.confidence).toUpperCase())} CONFIDENCE</span>` : ""}
      ${validation.degraded ? `<span class="meta-chip">DEGRADED</span>` : ""}
      ${validation.partially_supported ? `<span class="meta-chip">PARTIAL</span>` : ""}
      <span class="meta-chip">${sources.length} SOURCE${sources.length === 1 ? "" : "S"}</span>
      ${elapsed ? `<span class="meta-chip">${escapeHtml(seconds(elapsed))}</span>` : ""}
    </div>
    <article class="answer" id="answer">${body}</article>
    ${callout}
    ${
      assistant
        ? `<div class="answer-actions">
      <div class="action-group">
        <button class="mini-action" id="copyAnswer" type="button" aria-label="Copy answer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>
        <button class="mini-action ${assistant.rating === "helpful" ? "active" : ""}" id="helpful" type="button" aria-pressed="${assistant.rating === "helpful"}" aria-label="Helpful answer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M7 10v12H3V10h4zM7 20h10a2 2 0 0 0 2-1.6l1.4-7A2 2 0 0 0 18.4 9H14l1-4c.4-1.8-2-2.8-3-1.2L7 10z"/></svg></button>
        <button class="mini-action ${assistant.rating === "unhelpful" ? "active" : ""}" id="unhelpful" type="button" aria-pressed="${assistant.rating === "unhelpful"}" aria-label="Unhelpful answer"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M17 14V2h4v12h-4zM17 4H7a2 2 0 0 0-2 1.6l-1.4 7A2 2 0 0 0 5.6 15H10l-1 4c-.4 1.8 2 2.8 3 1.2l5-6.2z"/></svg></button>
      </div>
      <span class="verified">${validation.passed ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 12 2 2 4-5"/><circle cx="12" cy="12" r="9"/></svg> validation passed` : "not released as grounded"}</span>
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

const calloutCopy = {
  retrieval_abstain: ["No strong evidence", "Jev did not keep any passage as evidence for this question. The closest passages are listed as related, not as sources."],
  injection_scan: ["Passages excluded", "The matching passages contained text that reads like instructions to the assistant, so they were not used."],
  checker_ungrounded: ["Draft withheld", "The draft made claims the cited passages do not support."],
  checker_irrelevant: ["Draft withheld", "The draft did not answer the question that was asked."],
  checker_unsafe: ["Draft withheld", "The grounding check flagged the draft as unsafe."],
  checker_unparseable: ["Check unavailable", "The grounding check did not return a readable verdict, so the draft was withheld."],
  all_sentences_dropped: ["Draft withheld", "Too little of the draft was supported by the passages."],
  empty_draft: ["No draft", "The answer model returned an empty draft."],
};

function answerCallout(message, validation) {
  if (!message) return "";
  let title = "";
  let text = "";
  const category = String(validation.reject_category || "");
  if (validation.declined) {
    title = "The documents do not answer this";
    text = "The answer below says what the sources do and do not cover. Add a source if this question should be answerable.";
  } else if (validation.passed && validation.partially_supported) {
    title = "Trimmed to what the sources support";
    text = "Some sentences of the draft were removed because the passages did not support them.";
  } else if (!validation.passed && category.startsWith("deterministic:")) {
    title = "Draft withheld";
    text = validation.reason || "The draft cited a passage, number, or quotation that is not in the sources.";
  } else if (!validation.passed && calloutCopy[category]) {
    [title, text] = calloutCopy[category];
  } else if (!validation.passed && message.validation) {
    title = "Not released as grounded";
    text = validation.reason || "The draft did not pass the grounding, safety, and relevance check.";
  }
  if (validation.degraded) {
    title = title || "Reduced confidence";
    text = `${text ? `${text} ` : ""}Jev was unavailable, so passages were ranked by a fallback.`;
  }
  if (!title) return "";
  return `<aside class="callout"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 3 2 21h20L12 3z"/><path d="M12 9v5M12 18h.01"/></svg><div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p></div></aside>`;
}

function traceFlow(trace, validation) {
  const stages = [
    ["Query", Boolean(trace.original_query || trace.retrieval_query)],
    ["Search", trace.candidates != null],
    ["Rerank", Boolean(trace.rerank_mode)],
    ["Verify", Boolean(validation && validation.passed)],
  ];
  return `<div class="trace-flow">${stages
    .map(([label, done], index) => `${index ? `<i class="trace-line ${done ? "done" : ""}"></i>` : ""}<div class="trace-node ${done ? "done" : ""}"><i class="node-dot"></i><span>${label}</span></div>`)
    .join("")}</div>`;
}

function indexFoot() {
  const index = state.index || {};
  const chunks = state.documents.filter((doc) => doc.status === "indexed").reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);
  const healthy = index.compatible !== false;
  return `<div class="index-foot"><div class="index-line"><strong>Index health</strong><span>${healthy ? `${chunks} CHUNKS · ${escapeHtml(String(index.version_id || "").slice(0, 8))}` : "REFRESH NEEDED"}</span></div><div class="progress"><i style="width:${healthy ? 100 : 30}%"></i></div></div>`;
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
      <div class="trace-head"><span>Retrieval trace</span><span class="trace-time">${trace.latencies_ms?.total ? `${escapeHtml(seconds(trace.latencies_ms.total))} total` : `${trace.candidates ?? 0} candidates`}</span></div>
      ${traceFlow(trace, message.validation)}
      <div class="trace-stats">
        <div><strong>${trace.candidates ?? 0} → ${trace.kept ?? sources.length}</strong><span>chunks retained</span></div>
        <div><strong>${Number(trace.similarity_threshold ?? state.settings?.similarity_threshold ?? 0).toFixed(2)}</strong><span>min similarity</span></div>
        <div><strong>${escapeHtml(String(trace.version_id || state.index?.version_id || "").slice(0, 8) || "—")}</strong><span>index version</span></div>
      </div>
      <details><summary>View trace</summary><pre>${escapeHtml(JSON.stringify(trace, null, 2))}</pre></details>
    </section>`
    : "";
  body.innerHTML = `
    ${traceBlock}
    <div class="source-title"><strong>${sources.some((source) => source.relation === "related") ? "Related passages" : "Supporting sources"}</strong><span>${sources.length} MATCH${sources.length === 1 ? "" : "ES"}</span></div>
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
    </div>
    ${indexFoot()}`;
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
  root.innerHTML = `<p class="status-line">Searching the index…</p><h2 class="query-title">${escapeHtml(activeConversationTitle())}</h2><p class="user-prompt"><span class="section-eyebrow">You asked</span>${escapeHtml(query)}</p>`;
  const prior = state.messages.slice();
  try {
    const response = await api.chat({
      query,
      document_id: state.scopeId || null,
      conversation_id: state.conversationId,
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
  const monthAgo = Date.now() - 30 * 24 * 3600 * 1000;
  const recent = indexed.filter((doc) => new Date(doc.uploaded_at || 0).getTime() > monthAgo).length;
  const collections = new Set(indexed.map((doc) => doc.collection || "General")).size;
  const lastHandoff = state.index?.stats?.last_handoff_at;
  $("#page-knowledge").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Knowledge operations</div><h1>Knowledge base</h1><p>Curate the source material that can be searched, cited, and used for grounded answers.</p></div>
        <div class="page-actions">
          <button class="btn primary" id="uploadButton" type="button">Upload files</button>
        </div>
      </header>
      <div class="stat-grid">
        <article class="stat-card accent"><span>Total documents</span><div class="stat-value">${indexed.length}</div><small class="stat-delta">${recent ? `+${recent} this month` : "None added this month"}${processing.length ? ` · ${processing.length} processing` : ""}</small></article>
        <article class="stat-card"><span>Indexed chunks</span><div class="stat-value">${chunks.toLocaleString()}</div><small>Across ${collections} collection${collections === 1 ? "" : "s"}</small></article>
        <article class="stat-card"><span>Storage used</span><div class="stat-value">${bytes(storage)}</div><small>Original files kept locally</small></article>
        <article class="stat-card dark"><span>Index status</span><div class="stat-value">${state.index?.compatible === false ? "Refresh" : "Healthy"}</div><small>${state.index?.compatible === false ? "Embedding model changed" : lastHandoff ? `Last handoff ${escapeHtml(when(lastHandoff).toLowerCase())}` : escapeHtml(state.index?.embedding_model || "")}</small></article>
      </div>
      <section class="panel">
        <div class="panel-head">
          <div><h2>All documents</h2><p>Sources available to answers in this workspace</p></div>
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
          <table class="data-table"><thead><tr><th>Document</th><th>Collection</th><th>Chunks</th><th>Updated</th><th>Status</th><th></th></tr></thead>
          <tbody id="docRows"></tbody></table>
        </div>
      </section>
    </div>`;
  listen($("#uploadButton"), "click", () => chooseFile("upload"));
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
    body.innerHTML = `<tr><td colspan="6"><p class="empty-note">${note}</p></td></tr>`;
    return;
  }
  body.innerHTML = documents
    .map(
      (doc) => `<tr class="document-row" data-id="${escapeHtml(doc.id)}" data-status="${escapeHtml(doc.status || "")}">
        <td><div class="file-cell"><div class="doc-icon">${fileKind(doc.name)}</div><div><strong>${escapeHtml(doc.name)}</strong><span>${bytes(doc.bytes)} · ${doc.max_page || 0} page${doc.max_page === 1 ? "" : "s"}${doc.included === false ? " · excluded" : ""}</span></div></div></td>
        <td>${escapeHtml(doc.collection || "General")}</td>
        <td>${doc.chunk_count || "—"}</td>
        <td>${escapeHtml(doc.uploaded_at ? when(doc.uploaded_at) : "—")}</td>
        <td><span class="status ${doc.status === "indexed" ? "" : "sync"}">${escapeHtml(statusLabel(doc))}</span></td>
        <td><button class="row-action" type="button" aria-label="Open ${escapeHtml(doc.name)}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 18 6-6-6-6"/></svg></button></td>
      </tr>`
    )
    .join("");
  body.querySelectorAll(".document-row").forEach((row) => {
    listen(row, "click", () => openListedDocument(row.dataset.id));
  });
}

function statusLabel(doc) {
  if (doc.status === "indexed") return doc.included === false ? "Excluded" : "Indexed";
  return { queued: "Queued", processing: "Processing", failed: "Failed" }[doc.status] || String(doc.status || "Unknown");
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
  const passages = (doc.passages || []).slice(0, 12);
  page.innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div>
          <button class="btn small" id="backKnowledge" type="button">← Back to knowledge base</button>
          <h1 style="font-size:46px">${escapeHtml(doc.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " "))}</h1>
          <p>${escapeHtml(doc.name)} · ${escapeHtml(doc.collection || "General")} collection · ${doc.chunk_count || 0} chunks · ${doc.max_page || 0} page${doc.max_page === 1 ? "" : "s"}${doc.uploaded_at ? ` · added ${escapeHtml(when(doc.uploaded_at).toLowerCase())}` : ""}</p>
        </div>
        <div class="page-actions">
          <button class="btn" id="replaceFile" type="button">Replace file</button>
          <button class="btn dark" id="openOriginal" type="button" ${doc.downloadable && state.settings?.allow_downloads !== false ? "" : "disabled title=\"The original file is not available\""}>Open original ↗</button>
          <button class="btn danger" id="deleteDoc" type="button">Delete</button>
        </div>
      </header>
      <div class="doc-layout">
        <article class="panel document-preview" id="documentPreview">${documentPreview(doc.passages || [], doc.passage_count)}</article>
        <aside class="stack">
          <section class="panel">
            <div class="panel-head"><div><h2>Index record</h2><p>Saved with this document</p></div></div>
            <div class="form-block">
              <div class="switch-row"><div><strong>Included in answers</strong><p>When off, retrieval skips this file</p></div><button type="button" class="switch ${doc.included ? "on" : ""}" id="includedSwitch" role="switch" aria-pressed="${doc.included ? "true" : "false"}"></button></div>
              <div class="switch-row"><div><strong>Citation required</strong><p>Answers drawing on this file must cite a source</p></div><button type="button" class="switch ${doc.citation_required ? "on" : ""}" id="citeSwitch" role="switch" aria-pressed="${doc.citation_required ? "true" : "false"}"></button></div>
            </div>
          </section>
          <section class="panel">
            <div class="panel-head"><div><h2>Indexed passages</h2><p>${doc.chunk_count || 0} chunks · ${doc.passage_count ?? (doc.passages || []).length} parent passages · ${escapeHtml(state.index?.chunking || "Parent-child")}</p></div></div>
            <div class="chunk-list">${passages.map((item, index) => `<button type="button" class="chunk" data-passage="${index}"><span>PASSAGE ${String(index + 1).padStart(3, "0")} · PAGE ${item.page_number || "—"}${item.header_context ? ` · ${escapeHtml(item.header_context.split(" > ").pop())}` : ""}</span><p>${escapeHtml(String(item.text || "").replace(/^#{1,6}\s+/gm, "").replace(/\s+/g, " ").slice(0, 200))}…</p></button>`).join("") || `<p class="empty-note">No chunks stored.</p>`}</div>
          </section>
        </aside>
      </div>
    </div>`;
  listen($("#backKnowledge"), "click", () => showPage("knowledge"));
  page.querySelectorAll("[data-passage]").forEach((button) => {
    listen(button, "click", () => {
      page.querySelectorAll("[data-passage]").forEach((item) => item.classList.toggle("active", item === button));
      page.querySelectorAll("#documentPreview .preview-passage").forEach((item) => item.classList.remove("focus"));
      const target = page.querySelector(`#documentPreview [data-preview="${button.dataset.passage}"]`);
      if (!target) {
        notify("That passage is beyond the preview.");
        return;
      }
      target.classList.add("focus");
      target.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  });
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

// Render stored passages as a readable preview: Markdown-style headings become headings,
// tables and plain lines stay text. Each passage is addressable so a chunk can be focused.
function documentPreview(passages, total) {
  if (!passages.length) return `<p class="empty-note">No extractable preview.</p>`;
  let lastPage = null;
  const blocks = passages.map((passage, index) => {
    const lines = String(passage.text || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
    const body = lines
      .map((line) => {
        const heading = /^(#{1,4})\s+(.*)$/.exec(line);
        if (heading) return heading[1].length <= 1 ? `<h2>${escapeHtml(heading[2])}</h2>` : `<h3>${escapeHtml(heading[2])}</h3>`;
        if (/^\|.*\|$/.test(line)) return /^\|[\s|:-]+\|$/.test(line) ? "" : `<p class="preview-table">${escapeHtml(line)}</p>`;
        return `<p>${escapeHtml(line.replace(/^[-*]\s+/, "• "))}</p>`;
      })
      .join("");
    const pageMark = passage.page_number !== lastPage ? `<div class="section-eyebrow">Page ${escapeHtml(passage.page_number || "—")}</div>` : "";
    lastPage = passage.page_number;
    return `${pageMark}<section class="preview-passage" data-preview="${index}">${body}</section>`;
  });
  const more = total > passages.length ? `<p class="empty-note">Showing the first ${passages.length} of ${total} passages.</p>` : "";
  return blocks.join("") + more;
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
  const response = await api.documentFile(doc.id);
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(errorMessage(data, "The original file is not stored for this document."));
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const header = response.headers.get("Content-Disposition") || "";
  const named = /filename="?([^";]+)"?/i.exec(header);
  const isPdf = (blob.type || "").startsWith("application/pdf");
  const opened = isPdf ? window.open(url, "_blank", "noopener") : null;
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
  const [index, settings, evals] = await Promise.all([api.index(), ensureSettings(), api.evalLatest().catch(() => ({}))]);
  state.index = index;
  state.settings = settings;
  const stats = index.stats || {};
  const drift = settings.settings_drift || {};
  const driftKeys = Object.keys(drift);
  const driftNote = driftKeys.length
    ? `<div class="settings-drift-warning" role="status"><strong>Workspace settings differ from server defaults</strong><p>Questions use the saved workspace values. ${driftKeys
        .map((key) => `${escapeHtml(key)}: ${escapeHtml(drift[key].saved)} (server default ${escapeHtml(drift[key].env_default)})`)
        .join(" · ")}. The eval pins its own values in <code>backend/eval/config.json</code>.</p></div>`
    : "";
  const version = escapeHtml(String(index.version_id || "").slice(0, 8) || "—");
  const jevState = index.jev_circuit === "open" ? "Degraded" : index.jev_configured ? "Ready" : "No key";
  const latestEval = evals?.hermetic || null;
  const node = (label, title, detail, on = false) =>
    `<div class="pipe-node ${on ? "on" : ""}"><span class="node-label">${label}</span><strong>${title}</strong><small>${detail}</small></div>`;
  $("#page-pipeline").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Index architecture</div><h1>Pipeline</h1><p>Ingestion, retrieval, Jev reranking, and the grounding check for the active index.</p></div>
        <div class="page-actions">
          <button class="btn" id="viewRegistry" type="button">View registry</button>
          <button class="btn" id="reconcileIndex" type="button">Reconcile</button>
          <button class="btn primary" id="refreshIndex" type="button"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>Refresh index</button>
        </div>
      </header>
      ${driftNote}
      <div class="stat-grid">
        <article class="stat-card"><span>Active version</span><div class="stat-value">${version}</div><small>${escapeHtml(index.embedding_model || "")} · ${index.embedding_dimensions ?? "—"} dimensions</small></article>
        <article class="stat-card accent"><span>Pipeline health</span><div class="stat-value">${stats.run_success_rate == null ? "—" : percent(stats.run_success_rate)}</div><small>${stats.runs_counted ? `${stats.runs_counted} runs in 30 days` : "No runs in 30 days"}</small></article>
        <article class="stat-card"><span>Mean retrieval</span><div class="stat-value">${stats.mean_retrieval_ms == null ? "—" : escapeHtml(seconds(stats.mean_retrieval_ms))}</div><small>Search + rerank, last 30 days</small></article>
        <article class="stat-card dark"><span>Last handoff</span><div class="stat-value">${stats.last_handoff_at ? escapeHtml(when(stats.last_handoff_at)) : "—"}</div><small>${index.chunks ?? 0} chunks in the active index</small></article>
      </div>
      <div class="two-column">
        <section class="panel">
          <div class="panel-head"><div><h2>Live architecture</h2><p>What a question actually runs</p></div><span class="status ${index.compatible ? "" : "sync"}">${index.compatible ? "All systems normal" : "Needs refresh"}</span></div>
          <div class="pipeline-map">
            <div class="branch-label">Ingestion path</div>
            ${pipeRow([
              node("Source", "Documents", `${index.documents ?? 0} active`, true),
              node("Prepare", `${escapeHtml(index.chunking || "Parent-child")} chunking`, `${index.chunks ?? 0} chunks`, true),
              node("Encode", "Tokenize + embed", `${index.embedding_dimensions ?? "—"} dims · ${escapeHtml(index.embed_style || "raw")}`, true),
              node("Store", "Vector + keyword index", `${version} active`, true),
            ])}
            <div class="branch-label" style="margin-top:38px">Query path</div>
            ${pipeRow([
              node("Input", "Condense + embed", "follow-ups rewritten"),
              node("Retrieve", "Hybrid search", `k = ${settings.top_k} · floor ${Number(settings.similarity_threshold).toFixed(2)}`),
              node("Refine", "Jev reranker", `${escapeHtml(jevState)} · keep ≥ ${Number(index.jev_relevance_threshold ?? 0.2).toFixed(2)}`),
              node("Answer", "Ground + validate", latestEval?.metrics?.answers ? `${percent(latestEval.metrics.answers.answer_rate)} eval answer rate` : escapeHtml(index.answer_model || "OpenRouter")),
            ])}
            <div class="pipeline-legend"><span><i></i> Production active</span><span>● Runtime component</span><span>Jev via OpenRouter · ${escapeHtml(index.jev_model || "")}</span></div>
          </div>
        </section>
        <div class="stack">
          <section class="panel">
            <div class="panel-head"><div><h2>Recent runs</h2><p>Uploads, deletes, and handoffs</p></div></div>
            <div class="run-list">${(index.runs || []).slice(0, 6).map((run) => `<div class="run"><span class="run-time">${escapeHtml(clock(run.created_at))}</span><div><strong>${escapeHtml(run.name)}</strong><p>${escapeHtml(run.detail)}</p></div><span class="status ${run.status === "Success" ? "" : "sync"}">${escapeHtml(run.status)}</span></div>`).join("") || `<p class="empty-note">No runs recorded yet.</p>`}</div>
          </section>
          <section class="panel">
            <div class="panel-head"><div><h2>Thresholds</h2><p>Active retrieval configuration</p></div></div>
            <div class="metric-list">
              <div class="metric-row"><div><strong>Top-k retrieval</strong><p>Initial candidate pool</p></div><div class="metric-score">${settings.top_k}</div></div>
              <div class="metric-row"><div><strong>Similarity floor</strong><p>Minimum cosine score</p></div><div class="metric-score">${Number(settings.similarity_threshold).toFixed(2)}</div></div>
              <div class="metric-row"><div><strong>Jev relevance</strong><p>Passages below this are dropped</p></div><div class="metric-score">${Number(index.jev_relevance_threshold ?? 0.2).toFixed(2)}</div></div>
              <div class="metric-row"><div><strong>Rerank limit</strong><p>Context passages retained</p></div><div class="metric-score">${settings.max_parents}</div></div>
            </div>
          </section>
          <section class="panel">
            <div class="panel-head"><div><h2>Quality gate</h2><p>${latestEval ? `Hermetic eval · ${escapeHtml(latestEval.tier)} tier · ${escapeHtml(when(latestEval.finished_at))}` : "No eval run recorded yet"}</p></div>${latestEval ? `<span class="status ${latestEval.gate?.passed ? "" : "sync"}">${latestEval.gate?.passed ? "Pass" : "Fail"}</span>` : ""}</div>
            ${evalPanel(latestEval)}
          </section>
        </div>
      </div>
    </div>`;
  listen($("#viewRegistry"), "click", () => showRegistry());
  listen($("#reconcileIndex"), "click", async () => {
    const result = await api.reconcileIndex();
    const orphans = result.orphaned_documents || [];
    notify(
      `Reconcile removed ${result.removed} orphaned record${result.removed === 1 ? "" : "s"}` +
        (orphans.length ? `; ${orphans.length} indexed document${orphans.length === 1 ? " is" : "s are"} missing from the catalog` : "")
    );
    await renderPipeline();
  });
  listen($("#refreshIndex"), "click", (event) => refreshIndex(event.currentTarget));
}

function evalPanel(report) {
  if (!report) {
    return `<div class="form-block"><p>Run <code>python -m eval run</code> in <code>backend/</code> to measure retrieval and answer quality on the repo's test corpus.</p></div>`;
  }
  const retrieval = report.metrics?.retrieval || {};
  const answers = report.metrics?.answers;
  const abstain = report.metrics?.abstention;
  const rows = [
    ["Recall@5", "Expected passage in the top five", percent(retrieval.hit_at_5)],
    ["MRR", "Rank of the first relevant passage", retrieval.mrr == null ? "—" : Number(retrieval.mrr).toFixed(2)],
  ];
  if (answers) {
    rows.push(["Answer rate", "Answerable questions released as grounded", percent(answers.answer_rate)]);
    rows.push(["Key-fact recall", "Facts present in released answers", percent(answers.key_fact_recall)]);
    rows.push(["Abstention recall", "Unanswerable questions withheld", percent(abstain?.recall)]);
  }
  return `<div class="metric-list">${rows
    .map(([label, hint, value]) => `<div class="metric-row"><div><strong>${label}</strong><p>${hint}</p></div><div class="metric-score">${value}</div></div>`)
    .join("")}</div>`;
}

async function refreshIndex(button, override = false) {
  button.disabled = true;
  button.textContent = "Refreshing…";
  try {
    const status = await api.refreshIndex(override);
    notify(`Published index ${String(status.version_id || "").slice(0, 8)}`);
    await refreshShell();
    await renderPipeline();
  } catch (err) {
    button.disabled = false;
    button.textContent = "Refresh index";
    if (err instanceof ApiError && err.status === 409 && err.detail?.overridable) {
      openModal("Publish without the recall check?", `<p>${escapeHtml(err.message)}</p><p>The new version is checked for a complete copy either way. Only the golden-question recall comparison is skipped.</p>`, [
        { label: "Cancel", onClick: closeModal },
        {
          label: "Publish anyway",
          primary: true,
          onClick: async () => {
            closeModal();
            await refreshIndex(button, true);
          },
        },
      ]);
      return;
    }
    notify(err.message);
  }
}

async function showRegistry() {
  const { versions = [] } = await api.indexVersions();
  const rows = versions
    .map(
      (item) => `<tr><td><strong>${escapeHtml(String(item.version_id).slice(0, 8))}</strong><br><small>${escapeHtml(item.collection_name)}</small></td><td><span class="status ${item.status === "active" ? "" : "sync"}">${escapeHtml(item.status)}</span></td><td>${escapeHtml(item.embedding_model)}<br><small>${escapeHtml(item.embed_style)} · ${escapeHtml(item.chunking)}</small></td><td>${item.chunk_count ?? "—"}</td><td>${escapeHtml(when(item.activated_at || item.created_at))}</td></tr>`
    )
    .join("");
  const canRollBack = versions.some((item) => item.status === "retired");
  openModal(
    "Index registry",
    `<div class="table-scroll"><table class="data-table"><thead><tr><th>Version</th><th>Status</th><th>Model</th><th>Chunks</th><th>When</th></tr></thead><tbody>${rows || `<tr><td colspan="5"><p class="empty-note">No versions recorded.</p></td></tr>`}</tbody></table></div>`,
    [
      { label: "Close", onClick: closeModal },
      ...(canRollBack
        ? [
            {
              label: "Roll back to previous",
              danger: true,
              onClick: async () => {
                const restored = await api.rollbackIndex();
                closeModal();
                notify(`Restored index ${String(restored.version_id || "").slice(0, 8)}`);
                await refreshShell();
                await renderPipeline();
              },
            },
          ]
        : []),
    ],
    { wide: true }
  );
}

async function renderAnalytics() {
  const [report, evals] = await Promise.all([api.analytics(state.analyticsDays), api.evalLatest().catch(() => ({}))]);
  const series = report.series || [];
  const prior = report.prior || {};
  const max = Math.max(1, ...series.map((point) => Number(point.questions) || 0));
  const gaps = report.gaps || [];
  const step = Math.max(1, Math.round(series.length / 5));
  const labels = series.filter((_, index) => index % step === 0).map((point) => `<span>${escapeHtml(shortDay(point.day))}</span>`).join("");
  const perDay = report.questions ? report.questions / (report.days || state.analyticsDays) : 0;
  const dailyAverage = perDay && perDay < 0.1 ? "<0.1" : perDay.toFixed(1);
  const evalReport = evals?.hermetic;
  const abstainPrecision = evalReport?.metrics?.abstention?.precision;
  $("#page-analytics").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Quality intelligence</div><h1>Analytics</h1><p>Adoption, answer quality, and coverage gaps from questions this workspace has actually asked.</p></div>
        <div class="page-actions">
          <div class="segmented" id="range">${[7, 30, 90].map((days) => `<button type="button" data-days="${days}" class="${days === state.analyticsDays ? "active" : ""}">${days}D</button>`).join("")}</div>
          <button class="btn" id="exportReport" type="button">Export report</button>
        </div>
      </header>
      <div class="stat-grid">
        <article class="stat-card dark"><span>Questions asked</span><div class="stat-value">${(report.questions ?? 0).toLocaleString()}</div><small>${dailyAverage} daily average</small></article>
        <article class="stat-card accent"><span>Grounded answer rate</span><div class="stat-value">${percent(report.grounded_rate)}</div><small class="stat-delta">${delta(report.grounded_rate, prior.grounded_rate) || `${report.questions ?? 0} recorded`}</small></article>
        <article class="stat-card"><span>Helpful rating</span><div class="stat-value">${report.ratings ? percent(report.helpful_rate) : "—"}</div><small>${report.ratings ?? 0} response${report.ratings === 1 ? "" : "s"} rated</small></article>
        <article class="stat-card"><span>Withheld rate</span><div class="stat-value">${percent(report.withheld_rate)}</div><small class="stat-delta">${delta(report.withheld_rate, prior.withheld_rate, { invert: true }) || "No prior period"}</small></article>
      </div>
      <div class="two-column">
        <section class="panel">
          <div class="panel-head"><div><h2>Answer volume &amp; quality</h2><p>Daily questions with the grounded share as the lighter bar</p></div><div class="source-type"><i class="live-dot"></i> LIVE METRICS</div></div>
          ${report.questions
            ? `<div class="chart" aria-label="${state.analyticsDays} day answer volume bar chart">${series.map((point) => `<div class="bar-group" title="${escapeHtml(point.day)}: ${point.questions} asked, ${point.grounded} grounded"><i class="bar" style="height:${Math.round(((Number(point.questions) || 0) / max) * 100)}%"></i><i class="bar secondary" style="height:${Math.round(((Number(point.grounded) || 0) / max) * 100)}%"></i></div>`).join("")}</div><div class="chart-labels">${labels}</div>`
            : `<p class="empty-note">Ask a few questions to fill this chart.</p>`}
        </section>
        <section class="panel">
          <div class="panel-head"><div><h2>Quality signals</h2><p>How the system is performing</p></div></div>
          <div class="metric-list">
            <div class="metric-row"><div><strong>Citation coverage</strong><p>Released answers that cite a source</p></div><div class="metric-score">${percent(report.citation_coverage)}</div></div>
            <div class="metric-row"><div><strong>Average relevance</strong><p>Best Jev score per question</p></div><div class="metric-score">${report.mean_relevance == null ? "—" : Number(report.mean_relevance).toFixed(2)}</div></div>
            <div class="metric-row"><div><strong>Retrieval latency</strong><p>p50 search + rerank</p></div><div class="metric-score">${report.retrieval_p50_ms == null ? "—" : escapeHtml(seconds(report.retrieval_p50_ms))}</div></div>
            <div class="metric-row"><div><strong>Fallback precision</strong><p>${evalReport ? "Correct withholds in the last eval" : "Run the full eval to measure"}</p></div><div class="metric-score">${percent(abstainPrecision)}</div></div>
          </div>
        </section>
      </div>
      <section class="panel" style="margin-top:14px">
        <div class="panel-head"><div><h2>Knowledge gaps</h2><p>No coverage: nothing strong enough was found. Check failed: the draft was the problem.</p></div><button class="btn small" id="gapUpload" type="button">Add source</button></div>
        <div class="table-scroll"><table class="data-table"><thead><tr><th>Question</th><th>Why</th><th>Attempts</th><th>Best match</th><th>Last asked</th><th>Action</th></tr></thead><tbody>
          ${gaps.map((gap, index) => `<tr><td><strong>${escapeHtml(gap.query)}</strong></td><td>${gap.outcome === "check_failed" ? "Draft failed the check" : "No strong passage"}</td><td>${gap.attempts}</td><td>${gap.best_similarity == null ? "—" : `${Number(gap.best_similarity).toFixed(2)} similarity`}</td><td>${escapeHtml(when(gap.last_seen))}</td><td><button class="btn small" type="button" data-gap="${index}">${gap.outcome === "check_failed" ? "Ask again" : "Add source"}</button></td></tr>`).join("") || `<tr><td colspan="6"><p class="empty-note">No withheld questions in this range.</p></td></tr>`}
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
  document.querySelectorAll("#page-analytics [data-gap]").forEach((button) => {
    listen(button, "click", () => {
      const gap = gaps[Number(button.dataset.gap)];
      if (gap.outcome === "check_failed") {
        showPage("ask");
        $("#askInput").value = gap.query;
        $("#askInput").dispatchEvent(new Event("input"));
        $("#askInput").focus();
        return;
      }
      chooseFile("upload");
    });
  });
}

async function exportReport() {
  const response = await api.analyticsExport(state.analyticsDays);
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
  ["setChunking", "chunking"],
  ["setLength", "answer_length"],
  ["setCiteStyle", "citation_style"],
  ["setTopK", "top_k"],
  ["setSim", "similarity_threshold"],
  ["setParents", "max_parents"],
  ["setRrf", "rrf_k"],
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
    show_traces: Boolean(form.show_traces),
    allow_downloads: Boolean(form.allow_downloads),
    answer_length: form.answer_length || current.answer_length || "Balanced",
    citation_style: form.citation_style || current.citation_style || "Inline numbered",
    require_citations: Boolean(form.require_citations),
    withhold_ungrounded: Boolean(form.withhold_ungrounded),
    top_k: number("top_k", "Top-k candidates", 1, 100),
    similarity_threshold: number("similarity_threshold", "Similarity threshold", 0, 1),
    max_parents: number("max_parents", "Reranked context limit", 1, 12),
    rrf_k: number("rrf_k", "Fusion constant", 1, 200),
    contextual_embeddings: Boolean(form.contextual_embeddings),
    chunking: form.chunking || current.chunking || "Parent-child",
  };
}

async function renderSettings() {
  const [settings, session] = await Promise.all([api.settings(), api.session()]);
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
          <button type="button" data-settings="access">Access</button>
          <button type="button" data-settings="billing">Data</button>
        </nav>
        <div>
          <section class="panel settings-section active" data-section="general">
            <div class="panel-head"><div><h2>General</h2><p>Workspace identity and regional preferences</p></div></div>
            <div class="form-block"><h3>Workspace profile</h3><p>Shown in the top bar and on exported reports.</p><div class="field-grid">
              <div class="field"><label for="setName">Workspace name</label><input id="setName" /></div>
              <div class="field"><label for="setProfile">Your name</label><input id="setProfile" /></div>
              <div class="field"><label for="setCollection">Default collection</label><select id="setCollection">${optionList(["General", "Product", "Security", "Research"], state.form.default_collection)}</select></div>
            </div></div>
            <div class="form-block"><h3>Workspace preferences</h3><p>Shared behavior for everyone in this workspace.</p>
            <div class="switch-row"><div><strong>Show retrieval traces</strong><p>Show the retrieval and reranking trace next to each answer</p></div>${switches("show_traces")}</div>
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
            <div class="panel-head"><div><h2>Retrieval</h2><p>Used on the next question. Heading-aware embeddings apply when you refresh the index.</p></div></div>
            <div class="form-block"><div class="field-grid">
              <div class="field"><label for="setTopK">Top-k candidates</label><input id="setTopK" type="number" min="1" max="100" /></div>
              <div class="field"><label for="setSim">Similarity threshold</label><input id="setSim" type="number" min="0" max="1" step="0.01" /></div>
              <div class="field"><label for="setParents">Reranked context limit</label><input id="setParents" type="number" min="1" max="12" /></div>
              <div class="field"><label for="setRrf">Fusion constant</label><input id="setRrf" type="number" min="1" max="200" /></div>
              <div class="field"><label for="setChunking">Chunking strategy</label><select id="setChunking">${optionList(["Parent-child", "Fixed window", "Index card summary"], state.form.chunking)}</select></div>
            </div>
            <div class="subtle-note" style="margin-top:16px">Search settings apply to the next question. Chunking and heading-aware embeddings take effect when you refresh the index (Pipeline → Refresh index).</div>
            <div class="switch-row"><div><strong>Heading-aware embeddings</strong><p>Next index refresh embeds the section title with each passage</p></div>${switches("contextual_embeddings")}</div>
            </div>
          </section>
          <section class="panel settings-section" data-section="access">
            <div class="panel-head"><div><h2>Access</h2><p>${session.auth_disabled ? "Sign-in is turned off on this server" : "Signed in with the workspace access token"}</p></div><button class="btn small" id="signOut" type="button" ${session.auth_disabled ? "disabled" : ""}>Sign out</button></div>
            <div class="form-block"><h3>How access works</h3><p>Everyone who opens this workspace signs in with the same access token, set as <code>NEEDLE_ACCESS_TOKEN</code> on the server (or generated on first start and stored in the data folder). Sessions last 12 hours. To revoke every session, change the token and <code>NEEDLE_SESSION_SECRET</code>, then restart.</p></div>
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
  listen($("#resetWorkspace"), "click", () => resetDialog());
  listen($("#signOut"), "click", () => signOut());
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

function openModal(title, body, actions, { wide = false, locked = false } = {}) {
  $("#modal").querySelector(".modal-card").classList.toggle("wide", wide);
  $("#modal").dataset.locked = locked ? "true" : "";
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

function closeModal(force = false) {
  if ($("#modal").dataset.locked === "true" && !force) return;
  $("#modal").dataset.locked = "";
  $("#modal").hidden = true;
  $("#modalBody").innerHTML = "";
  $("#modalActions").innerHTML = "";
}

let signInOpen = false;

function showSignIn(message = "") {
  if (signInOpen) return;
  signInOpen = true;
  openModal(
    "Sign in to Needle",
    `<p>Enter the workspace access token. It is set as <code>NEEDLE_ACCESS_TOKEN</code> on the server, or printed in the server console on first start.</p>${message ? `<p class="form-error">${escapeHtml(message)}</p>` : ""}<div class="field"><label for="accessToken">Access token</label><input id="accessToken" type="password" autocomplete="current-password" /></div>`,
    [
      {
        label: "Sign in",
        primary: true,
        onClick: async () => {
          const token = $("#accessToken").value.trim();
          if (!token) return;
          try {
            await api.login(token);
          } catch (err) {
            signInOpen = false;
            closeModal(true);
            showSignIn(err.message);
            return;
          }
          signInOpen = false;
          closeModal(true);
          await boot();
        },
      },
    ],
    { locked: true }
  );
  listen($("#accessToken"), "keydown", (event) => {
    if (event.key === "Enter") $("#modalActions .btn.primary")?.click();
  });
}

async function signOut() {
  await api.logout();
  state.conversationId = null;
  state.messages = [];
  showSignIn();
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

// The server queues an upload and indexes it in the background; follow the job to the end.
async function waitForIndexing(queued) {
  if (queued.status === "duplicate") return { id: queued.id, name: queued.name, duplicate: true };
  notify(`Indexing ${queued.name}…`);
  await refreshShell();
  if (state.page === "knowledge") paintDocuments();
  const deadline = Date.now() + 30 * 60 * 1000;
  let delay = 1000;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, delay));
    delay = Math.min(delay * 1.5, 5000);
    const job = await api.job(queued.job_id);
    if (job.status === "completed") return { id: job.document_id, name: job.filename };
    if (job.status === "failed") throw new Error(job.error || `${job.filename} could not be indexed.`);
  }
  throw new Error(`${queued.name} is still indexing. It will appear in the knowledge base when it finishes.`);
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
  notify(`Uploading ${file.name}…`);
  try {
    const uploaded = await waitForIndexing(await api.upload(file));
    let message = uploaded.duplicate ? `${uploaded.name} is already indexed` : `${uploaded.name} indexed`;
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
listen($("#avatar"), "click", () =>
  openModal("Sign out?", "<p>You will need the access token to sign in again.</p>", [
    { label: "Cancel", onClick: () => closeModal() },
    { label: "Sign out", danger: true, onClick: async () => { closeModal(); await signOut(); } },
  ])
);
window.addEventListener("needle:auth-required", () => showSignIn("Your session ended. Sign in again."));
if (/Mac|iPhone|iPad/.test(navigator.platform || "")) $("#searchShortcut").textContent = "⌘K";
$("#sourceToggle").setAttribute("aria-expanded", "false");
$("#voiceButton").setAttribute("aria-pressed", "false");

async function boot() {
  const session = await api.session();
  if (!session.authenticated) {
    showSignIn();
    return;
  }
  await refreshShell();
  renderConversation();
  if (location.hash && location.hash !== "#ask") routeFromHash();
}

boot().catch((err) => notify(err.message));
