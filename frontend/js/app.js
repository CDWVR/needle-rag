import { api, ApiError, errorMessage, request } from "./api.js";
import { renderMarkdown, plainText, escapeHtml } from "./markdown.js";
import { outcomeBar, columnChart, lineChart, rankBars, wireTips } from "./charts.js";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const drawerMode = () => window.matchMedia("(max-width: 1180px)").matches;
const phone = () => window.matchMedia("(max-width: 820px)").matches;

const state = {
  page: "ask",
  settings: null,
  form: null,
  draftDirty: false,
  documents: [],
  conversations: [],
  conversationId: null,
  messages: [],
  activeTurn: null,
  revealTurn: null,
  pending: null,
  scopeId: null,
  analyticsDays: 30,
  docQuery: "",
  docFilter: "all",
  docSort: { key: "uploaded_at", dir: "desc" },
  index: null,
  busy: false,
  uploading: false,
  uploads: [],
  activeDocumentId: null,
  focusPassage: null,
  doc: null,
  role: "owner",
  demo: false,
  voiceSeconds: 15,
  deletedThreads: new Map(),
  knownThreads: null,
};

const isOwner = () => state.role === "owner";
const SAMPLE_QUESTIONS = [
  "What is the vanishing gradient problem in RNNs, and how is it mitigated?",
  "What is the difference between an LSTM and a GRU?",
  "What is the difference between global and local attention?",
  "How does reciprocal rank fusion combine BM25 and dense retrieval results?",
];
const ACCEPTED = /\.(pdf|txt|md|text|docx|pptx|xlsx|csv)$/i;
const RETRIEVAL_DEFAULTS = { top_k: 30, similarity_threshold: 0.3, max_parents: 5, rrf_k: 60 };

const pageLabels = {
  ask: "Ask",
  knowledge: "Knowledge base",
  document: "Document",
  pipeline: "Index pipeline",
  analytics: "Analytics",
  settings: "Settings",
};

// The answer is released only after the grounding check, so the wait is shown as real stages.
// Each step lists the server's status stages (backend/pipeline.py, _status) that belong to it.
const PROGRESS = [
  { label: "Search", stages: ["condense", "search", "retry"] },
  { label: "Rerank", stages: ["context"] },
  { label: "Write", stages: ["generate"] },
  { label: "Check", stages: ["validate"] },
];
const stepOf = (stage) => Math.max(0, PROGRESS.findIndex((step) => step.stages.includes(stage)));

const LOGO_SVG = `<svg viewBox="0 0 64 64" aria-hidden="true"><path class="lg-l" fill="currentColor" d="M10 18 24 4v52H10Z"/><path class="lg-r" fill="currentColor" d="M40 8h14v38L40 60Z"/><path class="lg-d" fill="currentColor" d="M24 48 40 32V16L24 32Z"/><path class="lg-f" d="M24 32 40 16v16Z"/><path class="lg-n" d="M40 16h-6l6 6Z"/></svg>`;
const ICON = {
  copy: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="9" y="9" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`,
  sources: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>`,
  retry: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 11a8 8 0 1 0-2.3 5.7M20 4v7h-7"/></svg>`,
  edit: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>`,
  up: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M7 10v12H3V10h4zM7 20h10a2 2 0 0 0 2-1.6l1.4-7A2 2 0 0 0 18.4 9H14l1-4c.4-1.8-2-2.8-3-1.2L7 10z"/></svg>`,
  down: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M17 14V2h4v12h-4zM17 4H7a2 2 0 0 0-2 1.6l-1.4 7A2 2 0 0 0 5.6 15H10l-1 4c-.4 1.8 2 2.8 3 1.2l5-6.2z"/></svg>`,
  more: `<svg viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="5" cy="12" r="1.8"/><circle cx="12" cy="12" r="1.8"/><circle cx="19" cy="12" r="1.8"/></svg>`,
  check: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 12 2 2 4-5"/><circle cx="12" cy="12" r="9"/></svg>`,
  warn: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 3 2 21h20L12 3z"/><path d="M12 9v5M12 18h.01"/></svg>`,
  eye: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>`,
  eyeOff: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 3l18 18M10.6 10.6a3 3 0 0 0 4.2 4.2M9.9 5.1A10 10 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.2M6.6 6.6A17 17 0 0 0 2 12s3.6 7 10 7a9.8 9.8 0 0 0 5.4-1.6"/></svg>`,
  trash: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 6h18M8 6V4h8v2M6 6l1 14h10l1-14"/></svg>`,
  chevron: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 18 6-6-6-6"/></svg>`,
  upload: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 16V4m0 0-5 5m5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/></svg>`,
};
const pipeArrow = `<div class="pipe-arrow" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M5 12h14m-5-5 5 5-5 5"/></svg></div>`;

// ---------------------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------------------

const initials = (name) =>
  String(name || "OP")
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() || "")
    .join("") || "OP";

const bytes = (size) => {
  const n = Number(size) || 0;
  if (!n) return "—";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
};

const asDate = (iso) => {
  const date = new Date(iso || "");
  return Number.isNaN(date.getTime()) ? null : date;
};
const isToday = (iso) => asDate(iso)?.toDateString() === new Date().toDateString();
const dayLabel = (date) =>
  date.toLocaleDateString(undefined, date.getFullYear() === new Date().getFullYear() ? { month: "short", day: "numeric" } : { month: "short", day: "numeric", year: "numeric" });
const timeLabel = (date) => date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });

// "Just now", "12 min ago", "3h ago", "Yesterday", "Oct 4", "Oct 4, 2025"
const when = (iso) => {
  const date = asDate(iso);
  if (!date) return "—";
  const mins = Math.round((Date.now() - date.getTime()) / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins} min ago`;
  if (isToday(iso)) return `${Math.round(mins / 60)}h ago`;
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return "Yesterday";
  return dayLabel(date);
};

// The same, mid-sentence: "added 3h ago", "added yesterday", "added on Oct 4".
const whenInline = (iso) => {
  const text = when(iso);
  return /^[A-Z][a-z]+ \d/.test(text) ? `on ${text}` : text.toLowerCase();
};

// A moment in a log: the time today, the date and time otherwise.
const stamp = (iso) => {
  const date = asDate(iso);
  if (!date) return "—";
  return isToday(iso) ? timeLabel(date) : `${dayLabel(date)} · ${timeLabel(date)}`;
};

const fileKind = (name) => {
  const ext = String(name || "").split(".").pop().toLowerCase();
  if (ext === "pdf") return "PDF";
  if (ext === "docx") return "DOC";
  if (ext === "csv" || ext === "xlsx") return "XLS";
  if (ext === "pptx") return "PPT";
  if (ext === "md") return "MD";
  return "TXT";
};

const docTitle = (name) => String(name || "Document").replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim();
const truncate = (text, n) => (String(text).length > n ? `${String(text).slice(0, n - 1).trimEnd()}…` : String(text));
const percent = (value) => (value == null ? "—" : `${Math.round(Number(value) * 1000) / 10}%`);
const seconds = (ms) => (ms == null ? "" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)}s`);
const shortDay = (iso) => {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleDateString([], { month: "short", day: "numeric" });
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

const linkify = (escaped) =>
  escaped.replace(/(https?:\/\/[^\s<]+[^\s<.,;:)!?'"])/g, '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>');

// ---------------------------------------------------------------------------------------
// Feedback: stacked toasts with optional actions
// ---------------------------------------------------------------------------------------

function notify(message, { type = "info", action = null, duration } = {}) {
  const stack = $("#toasts");
  const toast = document.createElement("div");
  const life = duration ?? (type === "error" ? 7000 : action ? 6000 : 3200);
  toast.className = `toast ${type}`;
  toast.setAttribute("role", type === "error" ? "alert" : "status");
  toast.style.setProperty("--life", `${life}ms`);
  toast.innerHTML = `<span class="toast-text"></span>${action ? `<button type="button" class="toast-action"></button>` : ""}<button type="button" class="toast-close" aria-label="Dismiss">×</button><i class="toast-life" aria-hidden="true"></i>`;
  $(".toast-text", toast).textContent = message;
  const dismiss = () => {
    if (toast.dataset.gone) return;
    toast.dataset.gone = "1";
    toast.classList.add("leaving");
    setTimeout(() => toast.remove(), reducedMotion() ? 0 : 200);
  };
  if (action) {
    const button = $(".toast-action", toast);
    button.textContent = action.label;
    button.addEventListener("click", () => {
      dismiss();
      Promise.resolve(action.onClick()).catch((err) => notify(err.message, { type: "error" }));
    });
  }
  $(".toast-close", toast).addEventListener("click", dismiss);
  // The life bar pauses on hover; when it runs out, the toast goes. Without animations, a timer does it.
  $(".toast-life", toast).addEventListener("animationend", dismiss);
  if (reducedMotion()) setTimeout(dismiss, life);
  stack.appendChild(toast);
  while (stack.children.length > 3) stack.firstElementChild.remove();
  return dismiss;
}

const listen = (node, type, fn) => {
  if (!node) return;
  node.addEventListener(type, (event) => {
    Promise.resolve(fn(event)).catch((err) => notify(err.message || "Something went wrong. Try again.", { type: "error" }));
  });
};

// ---------------------------------------------------------------------------------------
// Dialogs: focus moves in, stays in, and returns to what opened them
// ---------------------------------------------------------------------------------------

const dialog = { opener: null, onClose: null };
const focusables = (root) =>
  $$('a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])', root).filter(
    (node) => node.offsetParent !== null
  );

function openModal(title, body, actions, { wide = false, locked = false, onClose = null } = {}) {
  const root = $("#modal");
  if (root.hidden) dialog.opener = document.activeElement;
  dialog.onClose = onClose;
  $(".modal-card", root).classList.toggle("wide", wide);
  root.dataset.locked = locked ? "true" : "";
  $("#modalTitle").textContent = title;
  $("#modalBody").innerHTML = body;
  const bar = $("#modalActions");
  bar.innerHTML = "";
  actions.forEach((action) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `btn ${action.primary ? "primary" : ""} ${action.danger ? "danger" : ""}`;
    button.textContent = action.label;
    if (action.id) button.id = action.id;
    if (action.disabled) button.disabled = true;
    button.addEventListener("click", () => {
      Promise.resolve(action.onClick()).catch((err) => notify(err.message || "Something went wrong. Try again.", { type: "error" }));
    });
    bar.appendChild(button);
  });
  root.hidden = false;
  root.classList.remove("closing");
  requestAnimationFrame(() => root.classList.add("open"));
  const first = $("input, select, textarea", $("#modalBody")) || $(".btn.primary:not([disabled])", bar) || $(".btn", bar);
  first?.focus();
}

function closeModal(force = false) {
  const root = $("#modal");
  if (root.hidden) return;
  if (root.dataset.locked === "true" && !force) return;
  root.dataset.locked = "";
  root.classList.remove("open");
  root.classList.add("closing");
  const finish = () => {
    if (root.classList.contains("open")) return; // reopened while closing
    root.hidden = true;
    root.classList.remove("closing");
    $("#modalBody").innerHTML = "";
    $("#modalActions").innerHTML = "";
    const opener = dialog.opener;
    const onClose = dialog.onClose;
    dialog.opener = null;
    dialog.onClose = null;
    if (opener && document.contains(opener)) opener.focus();
    onClose?.();
  };
  if (reducedMotion()) finish();
  else setTimeout(finish, 150);
}

$("#modal").addEventListener("keydown", (event) => {
  if (event.key !== "Tab") return;
  const items = focusables($(".modal-card"));
  if (!items.length) return;
  const first = items[0];
  const last = items[items.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
});

// Resolves true when confirmed. `typed` asks the person to type a word before the button enables.
function confirmDialog({ title, body, confirmLabel = "Confirm", danger = false, typed = "" }) {
  return new Promise((resolve) => {
    let settled = false;
    const done = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    const field = typed
      ? `<div class="field"><label for="confirmTyped">Type <strong>${escapeHtml(typed)}</strong> to confirm</label><input id="confirmTyped" autocomplete="off" spellcheck="false" /></div>`
      : "";
    openModal(
      title,
      `<p>${body}</p>${field}`,
      [
        { label: "Cancel", onClick: () => { closeModal(); done(false); } },
        { label: confirmLabel, id: "confirmGo", primary: !danger, danger, disabled: Boolean(typed), onClick: () => { done(true); closeModal(); } },
      ],
      { onClose: () => done(false) }
    );
    if (typed) {
      const input = $("#confirmTyped");
      input.addEventListener("input", () => {
        $("#confirmGo").disabled = input.value.trim() !== typed;
      });
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !$("#confirmGo").disabled) $("#confirmGo").click();
      });
    }
  });
}

// ---------------------------------------------------------------------------------------
// Small popover menus (thread options, account)
// ---------------------------------------------------------------------------------------

let openMenuState = null;

function openMenu(anchor, items) {
  closeMenu();
  const menu = document.createElement("div");
  menu.className = "menu-pop";
  menu.setAttribute("role", "menu");
  items.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("role", "menuitem");
    if (item.danger) button.className = "danger";
    button.textContent = item.label;
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      closeMenu();
      Promise.resolve(item.onClick()).catch((err) => notify(err.message, { type: "error" }));
    });
    menu.appendChild(button);
  });
  document.body.appendChild(menu);
  const rect = anchor.getBoundingClientRect();
  const top = Math.min(rect.bottom + 6, window.innerHeight - menu.offsetHeight - 8);
  menu.style.top = `${Math.max(8, top)}px`;
  menu.style.left = `${Math.max(8, Math.min(rect.right - menu.offsetWidth, window.innerWidth - menu.offsetWidth - 8))}px`;
  anchor.setAttribute("aria-expanded", "true");
  openMenuState = { menu, anchor };
  $("button", menu)?.focus();
  menu.addEventListener("keydown", (event) => {
    const buttons = $$("button", menu);
    const index = buttons.indexOf(document.activeElement);
    if (event.key === "ArrowDown") {
      event.preventDefault();
      buttons[(index + 1) % buttons.length].focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      buttons[(index - 1 + buttons.length) % buttons.length].focus();
    }
  });
}

function closeMenu({ restoreFocus = false } = {}) {
  if (!openMenuState) return false;
  const { menu, anchor } = openMenuState;
  openMenuState = null;
  menu.remove();
  anchor.setAttribute("aria-expanded", "false");
  if (restoreFocus) anchor.focus();
  return true;
}

// ---------------------------------------------------------------------------------------
// Motion helpers
// ---------------------------------------------------------------------------------------

// Numbers on stat cards count up the first time they're shown.
function countUp(root) {
  $$("[data-count]", root).forEach((node) => {
    const target = Number(node.dataset.count);
    if (!Number.isFinite(target)) return;
    const format = node.dataset.format || "int";
    const paint = (value) => {
      node.textContent = format === "pct" ? `${Math.round(value * 10) / 10}%` : Math.round(value).toLocaleString();
    };
    if (reducedMotion() || target === 0) {
      paint(target);
      return;
    }
    const start = performance.now();
    const tick = (now) => {
      const k = Math.min(1, (now - start) / 650);
      paint(target * (1 - (1 - k) ** 3));
      if (k < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

// The Needle mark assembling and folding in a loop: the loading symbol everywhere something waits.
const loader = (size = "") => `<span class="needle-loader ${size}" aria-hidden="true">${LOGO_SVG}</span>`;

// A pill slides between the options of a segmented control. Analytics re-renders its control on
// each click, so the last position is remembered per control and the new pill starts from there.
const segmentedFrom = new Map();
function initSegmented(group) {
  if (!group) return;
  group.classList.add("has-pill");
  let pill = $(".seg-pill", group);
  if (!pill) {
    pill = document.createElement("i");
    pill.className = "seg-pill";
    pill.setAttribute("aria-hidden", "true");
    group.prepend(pill);
  }
  const put = ({ x, w }) => {
    pill.style.transform = `translateX(${x}px)`;
    pill.style.width = `${w}px`;
  };
  const place = (animate) => {
    const on = $("button.active", group);
    if (!on || !group.offsetParent) return;
    pill.classList.toggle("still", !animate);
    const at = { x: on.offsetLeft, w: on.offsetWidth };
    put(at);
    segmentedFrom.set(group.id, at);
  };
  const from = segmentedFrom.get(group.id);
  if (from) {
    pill.classList.add("still");
    put(from);
    void pill.offsetWidth; // commit the old position so the move animates
    place(true);
  } else place(false);
  const watch = new MutationObserver(() => place(true));
  $$("button", group).forEach((button) => watch.observe(button, { attributeFilter: ["class"] }));
  document.fonts?.ready.then(() => place(false));
}

// The rail's active highlight is one shape that slides to the page you open.
function moveRailPill(animate = true) {
  const rail = $(".rail");
  let pill = $(".rail-pill", rail);
  if (!pill) {
    pill = document.createElement("i");
    pill.className = "rail-pill gone";
    pill.setAttribute("aria-hidden", "true");
    rail.prepend(pill);
  }
  const on = $(".icon-btn.active", rail);
  if (!on || !on.offsetParent) {
    pill.classList.add("gone");
    rail.classList.remove("has-pill");
    return;
  }
  rail.classList.add("has-pill");
  const a = rail.getBoundingClientRect();
  const b = on.getBoundingClientRect();
  pill.classList.toggle("still", !animate || pill.classList.contains("gone"));
  pill.classList.remove("gone");
  pill.style.transform = `translate(${b.left - a.left}px, ${b.top - a.top}px)`;
  pill.style.width = `${b.width}px`;
  pill.style.height = `${b.height}px`;
}

// Native selects keep the value and the form wiring; this draws a styled button and option list
// over each one, with the keyboard behavior of a listbox.
function enhanceSelect(select) {
  if (select.dataset.enhanced) return;
  select.dataset.enhanced = "1";
  const wrap = document.createElement("div");
  wrap.className = "select";
  select.after(wrap);
  wrap.append(select);
  select.tabIndex = -1;
  select.setAttribute("aria-hidden", "true");
  const button = document.createElement("button");
  button.type = "button";
  button.className = "select-button";
  button.id = `${select.id}Button`;
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");
  const list = document.createElement("ul");
  list.className = "select-list";
  list.id = `${select.id}List`;
  list.setAttribute("role", "listbox");
  list.hidden = true;
  button.setAttribute("aria-controls", list.id);
  const label = $(`label[for="${select.id}"]`);
  if (label) {
    label.htmlFor = button.id;
    label.id ||= `${select.id}Label`;
    list.setAttribute("aria-labelledby", label.id);
    button.setAttribute("aria-labelledby", `${label.id} ${button.id}`);
  }
  wrap.append(button, list);
  let active = 0;
  const isOpen = () => !list.hidden;
  const paint = () => {
    button.innerHTML = `<span>${escapeHtml(select.selectedOptions[0]?.text || "")}</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>`;
    list.innerHTML = [...select.options]
      .map(
        (option, i) =>
          `<li role="option" id="${list.id}-${i}" data-index="${i}" aria-selected="${option.selected}" class="${i === active ? "active" : ""}"><span>${escapeHtml(option.text)}</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" aria-hidden="true"><path d="m5 12 5 5 9-10"/></svg></li>`
      )
      .join("");
    if (isOpen()) button.setAttribute("aria-activedescendant", `${list.id}-${active}`);
    else button.removeAttribute("aria-activedescendant");
  };
  const open = (yes) => {
    if (yes) active = Math.max(0, select.selectedIndex);
    list.hidden = !yes;
    wrap.classList.toggle("open", yes);
    button.setAttribute("aria-expanded", String(yes));
    paint();
  };
  const choose = (i) => {
    if (select.selectedIndex !== i) {
      select.selectedIndex = i;
      select.dispatchEvent(new Event("input", { bubbles: true }));
      select.dispatchEvent(new Event("change", { bubbles: true }));
    }
    open(false);
  };
  const move = (i) => {
    active = Math.min(Math.max(i, 0), select.options.length - 1);
    paint();
  };
  button.addEventListener("click", () => open(!isOpen()));
  button.addEventListener("blur", () => isOpen() && open(false));
  button.addEventListener("keydown", (event) => {
    const keys = {
      ArrowDown: () => (isOpen() ? move(active + 1) : open(true)),
      ArrowUp: () => (isOpen() ? move(active - 1) : open(true)),
      Home: () => isOpen() && move(0),
      End: () => isOpen() && move(select.options.length - 1),
      Enter: () => (isOpen() ? choose(active) : open(true)),
      " ": () => (isOpen() ? choose(active) : open(true)),
      Escape: () => isOpen() && open(false),
    };
    if (!keys[event.key] || (event.key === "Escape" && !isOpen())) return;
    event.preventDefault();
    event.stopPropagation();
    keys[event.key]();
  });
  list.addEventListener("mousedown", (event) => event.preventDefault()); // keep focus on the button
  list.addEventListener("click", (event) => {
    const item = event.target.closest("li");
    if (item) choose(Number(item.dataset.index));
  });
  list.addEventListener("mousemove", (event) => {
    const item = event.target.closest("li");
    if (item && Number(item.dataset.index) !== active) move(Number(item.dataset.index));
  });
  select.refreshCustom = paint;
  paint();
}

const skeletonPage = () => `
  <div class="page-content skeleton-page" aria-hidden="true">
    <div class="sk-loader">${loader("large")}</div>
    <div class="sk sk-eyebrow"></div><div class="sk sk-title"></div><div class="sk sk-line"></div>
    <div class="stat-grid">${'<div class="sk sk-card"></div>'.repeat(4)}</div>
    <div class="sk sk-panel"></div>
  </div>`;

// The first visit to a page shows a skeleton while it loads; later visits keep the old view until the new one is ready.
async function withSkeleton(page, render) {
  const node = $(`#page-${page}`);
  if (!node.childElementCount) node.innerHTML = skeletonPage();
  node.setAttribute("aria-busy", "true");
  try {
    await render();
  } finally {
    node.setAttribute("aria-busy", "false");
  }
}

// ---------------------------------------------------------------------------------------
// Shell: pages, drawers, title
// ---------------------------------------------------------------------------------------

function setTitle(text) {
  document.title = `${text} · ${state.settings?.workspace_name || "Needle"}`;
}

function syncScrim() {
  const inspectorOpen = $("#inspector").classList.contains("open") && drawerMode() && state.page === "ask";
  const sidebarOpen = $("#sidebar").classList.contains("open") && phone();
  $("#scrim").hidden = !(inspectorOpen || sidebarOpen);
}

function setSidebar(open) {
  $("#sidebar").classList.toggle("open", open);
  $("#threadsToggle").setAttribute("aria-expanded", open ? "true" : "false");
  syncScrim();
}

function setInspector(open) {
  $("#inspector").classList.toggle("open", open);
  $("#sourceToggle").setAttribute("aria-expanded", open ? "true" : "false");
  syncScrim();
}

let leaving = false;

async function showPage(page, { updateHash = true, force = false } = {}) {
  if (!pageLabels[page] || (page === "settings" && !isOwner())) {
    page = "ask";
    updateHash = true; // do not leave a page the viewer cannot open in the address bar
  }
  if (state.page === "settings" && page !== "settings" && state.draftDirty && !force) {
    if (leaving) return;
    leaving = true;
    const discard = await confirmDialog({
      title: "Discard unsaved settings?",
      body: "You changed settings on this page but haven't saved them.",
      confirmLabel: "Discard changes",
      danger: true,
    });
    leaving = false;
    if (!discard) {
      if (location.hash !== "#settings") history.pushState(null, "", "#settings");
      return;
    }
    state.draftDirty = false;
    state.form = { ...(state.settings || {}) };
  }
  state.page = page;
  if (updateHash) {
    const hash =
      page === "document" && state.activeDocumentId
        ? `#document/${state.activeDocumentId}${state.focusPassage ? `/${encodeURIComponent(state.focusPassage)}` : ""}`
        : `#${page}`;
    if (location.hash !== hash) history.pushState(null, "", hash);
  }
  $("#app").classList.toggle("subpage", page !== "ask");
  $$(".workspace-page").forEach((view) => view.classList.toggle("active", view.dataset.view === page));
  $$(".rail [data-page]").forEach((button) => {
    const on = button.dataset.page === page || (page === "document" && button.dataset.page === "knowledge");
    button.classList.toggle("active", on && !button.classList.contains("mark"));
    if (on) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  });
  moveRailPill();
  $("#currentCrumb").textContent = pageLabels[page] || "Ask";
  setTitle(page === "ask" ? activeConversationTitle() : pageLabels[page]);
  setSidebar(false);
  closeContextMenu();
  closeMenu();
  if (page !== "ask") setInspector(false);
  const renderers = { knowledge: renderKnowledge, document: renderDocument, pipeline: renderPipeline, analytics: renderAnalytics, settings: renderSettings };
  if (renderers[page]) withSkeleton(page, renderers[page]).catch((err) => notify(err.message, { type: "error" }));
  $(`#page-${page}`)?.scrollTo(0, 0);
}

function routeFromHash() {
  const [page, id, passage] = location.hash.replace(/^#/, "").split("/");
  if (page === "document" && id) {
    state.activeDocumentId = decodeURIComponent(id);
    state.focusPassage = passage ? decodeURIComponent(passage) : null;
  }
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
  const scopeStillThere = state.documents.some((doc) => doc.id === state.scopeId && doc.status === "indexed" && doc.included !== false);
  if (state.scopeId && !scopeStillThere) state.scopeId = null;
  updateScopeLabel();
  $("#workspaceName").textContent = settings.workspace_name;
  $("#avatar").textContent = initials(settings.profile_name);
  $("#avatar").setAttribute("aria-label", `Account: ${settings.profile_name}`);
  const version = String(index.version_id || "index").slice(0, 8);
  $("#syncLabel").textContent = `INDEX LIVE · ${version}`;
  if (state.page === "ask") setTitle(activeConversationTitle());
  renderThreads();
  renderCorpus();
}

// ---------------------------------------------------------------------------------------
// Conversations list
// ---------------------------------------------------------------------------------------

function renderThreads() {
  const list = $("#threadList");
  const items = uniqueById(state.conversations).filter((thread) => !state.deletedThreads.has(thread.id));
  const search = $("#threadSearch").value.trim();
  $("#threadCount").textContent = `${items.length} thread${items.length === 1 ? "" : "s"}`;
  if (!items.length) {
    list.innerHTML = search
      ? `<p class="empty-note">No conversations match “${escapeHtml(search)}”.</p>`
      : `<p class="empty-note">No conversations yet. Ask a question to start one.</p>`;
    return;
  }
  const known = state.knownThreads;
  const row = (thread) => {
    const sourced = Number(thread.source_threads) || 0;
    const meta = [when(thread.updated_at), sourced ? `${sourced} sourced` : ""].filter(Boolean).join(" · ");
    const fresh = known && !known.has(thread.id) ? "enter" : "";
    const active = thread.id === state.conversationId ? "active" : "";
    return `
      <div class="thread-row ${active} ${fresh}" data-id="${escapeHtml(thread.id)}">
        <button type="button" class="thread" data-id="${escapeHtml(thread.id)}" ${active ? 'aria-current="true"' : ""}>
          <strong>${escapeHtml(thread.title)}</strong>
          <small>${escapeHtml(meta)}</small>
        </button>
        <button type="button" class="thread-more" data-id="${escapeHtml(thread.id)}" aria-label="Options for ${escapeHtml(thread.title)}" aria-haspopup="menu" aria-expanded="false">${ICON.more}</button>
      </div>`;
  };
  const today = items.filter((thread) => isToday(thread.updated_at));
  const earlier = items.filter((thread) => !isToday(thread.updated_at));
  const group = (label, threads) =>
    threads.length ? `<div class="thread-label"><span class="section-eyebrow">${label}</span></div>${threads.map(row).join("")}` : "";
  list.innerHTML = group("Today", today) + group("Earlier", earlier);
  state.knownThreads = new Set(items.map((thread) => thread.id));
  $$(".thread", list).forEach((button) => listen(button, "click", () => openConversation(button.dataset.id)));
  $$(".thread-more", list).forEach((button) =>
    listen(button, "click", (event) => {
      event.stopPropagation();
      threadMenu(button, button.dataset.id);
    })
  );
}

function threadMenu(anchor, id) {
  openMenu(anchor, [
    { label: "Rename", onClick: () => renameThread(id) },
    { label: "Delete", danger: true, onClick: () => deleteThread(id) },
  ]);
}

function renameThread(id) {
  const thread = state.conversations.find((item) => item.id === id);
  const row = $(`.thread-row[data-id="${CSS.escape(id)}"]`);
  if (!thread) return;
  if (!row || row.offsetParent === null) {
    // The sidebar is hidden (phone, or a narrow window): rename in a dialog instead.
    return renameInDialog(thread);
  }
  const input = document.createElement("input");
  input.className = "thread-rename";
  input.value = thread.title;
  input.maxLength = 120;
  input.setAttribute("aria-label", "Conversation name");
  $(".thread", row).replaceWith(input);
  input.focus();
  input.select();
  let finished = false;
  const finish = async (save) => {
    if (finished) return;
    finished = true;
    const title = input.value.trim();
    if (!save || !title || title === thread.title) {
      renderThreads();
      return;
    }
    await saveTitle(thread, title);
  };
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      finish(true);
    } else if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      finish(false);
    }
  });
  input.addEventListener("blur", () => finish(true));
}

async function saveTitle(thread, title) {
  try {
    const renamed = await api.renameConversation(thread.id, title);
    thread.title = renamed.title;
    renderThreads();
    if (thread.id === state.conversationId) renderConversation();
    notify("Conversation renamed", { type: "success" });
  } catch (err) {
    renderThreads();
    notify(err.message, { type: "error" });
  }
}

function renameInDialog(thread) {
  openModal(
    "Rename conversation",
    `<div class="field"><label for="renameField">Name</label><input id="renameField" maxlength="120" value="${escapeHtml(thread.title)}" /></div>`,
    [
      { label: "Cancel", onClick: () => closeModal() },
      {
        label: "Save",
        primary: true,
        onClick: async () => {
          const title = $("#renameField").value.trim();
          if (!title) return;
          closeModal();
          await saveTitle(thread, title);
        },
      },
    ]
  );
  $("#renameField").addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("#modalActions .btn.primary").click();
  });
}

// Deleting waits a few seconds so Undo is possible; leaving the page sends it right away.
function deleteThread(id) {
  const thread = state.conversations.find((item) => item.id === id);
  const title = thread?.title || "Conversation";
  const wasActive = id === state.conversationId;
  $(`.thread-row[data-id="${CSS.escape(id)}"]`)?.classList.add("leaving");
  const commit = () =>
    api
      .deleteConversation(id)
      .then(() => {
        state.conversations = state.conversations.filter((item) => item.id !== id);
        state.deletedThreads.delete(id);
      })
      .catch((err) => {
        state.deletedThreads.delete(id);
        renderThreads();
        notify(err.message, { type: "error" });
      });
  const timer = setTimeout(commit, 6000);
  state.deletedThreads.set(id, timer);
  setTimeout(renderThreads, reducedMotion() ? 0 : 220);
  if (wasActive) startNewConversation({ focus: false });
  notify(`Deleted “${truncate(title, 42)}”`, {
    duration: 6000,
    action: {
      label: "Undo",
      onClick: () => {
        clearTimeout(timer);
        state.deletedThreads.delete(id);
        renderThreads();
        if (wasActive) return openConversation(id);
      },
    },
  });
}

window.addEventListener("pagehide", () => {
  for (const [id, timer] of state.deletedThreads) {
    clearTimeout(timer);
    request(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE", keepalive: true }).catch(() => {});
  }
  state.deletedThreads.clear();
});

function renderCorpus() {
  const indexed = state.documents.filter((doc) => doc.status === "indexed").slice(0, 4);
  $("#corpusList").innerHTML = indexed.length
    ? indexed
        .map(
          (doc) => `
        <button type="button" class="doc-stat" data-doc="${escapeHtml(doc.id)}">
          <div class="doc-icon">${fileKind(doc.name)}</div>
          <div class="doc-meta"><strong>${escapeHtml(doc.name)}</strong><span>${doc.chunk_count} chunks${doc.included === false ? " · excluded" : ""}</span></div>
          <i class="doc-ok ${doc.included === false ? "off" : ""}"></i>
        </button>`
        )
        .join("")
    : `<p class="empty-note">No documents indexed yet.</p>`;
  $$("[data-doc]", $("#corpusList")).forEach((button) => listen(button, "click", () => openDocument(button.dataset.doc)));
}

// ---------------------------------------------------------------------------------------
// Ask: the conversation as a thread of turns
// ---------------------------------------------------------------------------------------

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
  if (!state.conversationId) return "Ask";
  const thread = (state.conversations || []).find((item) => item.id === state.conversationId);
  return String(thread?.title || "").trim() || "Ask";
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
    text = "The answer says what the sources do and do not cover. Add a source if this question should be answerable.";
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
  return `<aside class="callout">${ICON.warn}<div><strong>${escapeHtml(title)}</strong><p>${escapeHtml(text)}</p></div></aside>`;
}

const citedNumbers = (content) => {
  const set = new Set();
  for (const match of String(content || "").matchAll(/\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]/g)) {
    match[1].split(",").forEach((n) => set.add(Number(n.trim())));
  }
  return set;
};

function verdict(answer) {
  const validation = answer?.validation || {};
  if (!answer) return { kicker: "Not answered", chip: "NO ANSWER", good: false };
  if (validation.declined) return { kicker: "Not in your sources", chip: "NOT IN SOURCES", good: false };
  if (validation.passed) return { kicker: "Grounded answer", chip: "GROUNDED", good: true };
  if (!answer.validation) return { kicker: "Answer", chip: "UNCHECKED", good: false };
  return { kicker: "Not released as grounded", chip: validation.reject_category === "retrieval_abstain" ? "NO EVIDENCE" : "CHECK FAILED", good: false };
}

function turnHtml(turn, index) {
  const { question, answer } = turn;
  const validation = answer?.validation || {};
  const sources = answer?.sources || [];
  const elapsed = answer?.trace?.latencies_ms?.total;
  const status = verdict(answer);
  const classes = ["turn", index === state.activeTurn ? "active" : "", index === state.revealTurn ? "reveal" : ""].join(" ");
  const body = answer
    ? `
      <div class="meta-line">
        <span class="meta-chip ${status.good ? "good" : ""}">${status.chip}</span>
        ${validation.confidence ? `<span class="meta-chip">${escapeHtml(String(validation.confidence).toUpperCase())} CONFIDENCE</span>` : ""}
        ${validation.degraded ? `<span class="meta-chip">DEGRADED</span>` : ""}
        ${validation.partially_supported ? `<span class="meta-chip">PARTIAL</span>` : ""}
        <button type="button" class="meta-chip chip-button" data-act="evidence">${sources.length} SOURCE${sources.length === 1 ? "" : "S"}</button>
        ${elapsed ? `<span class="meta-chip">${escapeHtml(seconds(elapsed))}</span>` : ""}
      </div>
      <article class="answer">${renderMarkdown(answer.content || "")}</article>
      ${answerCallout(answer, validation)}
      <div class="answer-actions">
        <div class="action-group">
          <button class="mini-action" type="button" data-act="copy" aria-label="Copy answer" data-tip="Copy">${ICON.copy}</button>
          <button class="mini-action" type="button" data-act="copy-sources" aria-label="Copy answer with sources" data-tip="Copy with sources">${ICON.sources}</button>
          ${question ? `<button class="mini-action" type="button" data-act="retry" aria-label="Ask this again" data-tip="Ask again">${ICON.retry}</button>
          <button class="mini-action" type="button" data-act="edit" aria-label="Edit question" data-tip="Edit question">${ICON.edit}</button>` : ""}
          <span class="action-divider" aria-hidden="true"></span>
          <button class="mini-action ${answer.rating === "helpful" ? "active" : ""}" type="button" data-act="helpful" aria-pressed="${answer.rating === "helpful"}" aria-label="Helpful answer" data-tip="Helpful">${ICON.up}</button>
          <button class="mini-action ${answer.rating === "unhelpful" ? "active" : ""}" type="button" data-act="unhelpful" aria-pressed="${answer.rating === "unhelpful"}" aria-label="Unhelpful answer" data-tip="Not helpful">${ICON.down}</button>
        </div>
        <span class="verified ${validation.passed ? "" : "muted"}">${validation.passed ? `${ICON.check} Validation passed` : "Not released as grounded"}</span>
      </div>
      <div class="feedback-reasons" hidden>
        <span>What went wrong?</span>
        <button type="button" class="chip-option" data-reason="wrong_source">Wrong source</button>
        <button type="button" class="chip-option" data-reason="incomplete">Incomplete</button>
        <button type="button" class="chip-option" data-reason="should_answer">Should have answered</button>
        <button type="button" class="chip-option" data-reason="other">Something else</button>
      </div>`
    : `<p class="empty-note stopped">No answer was saved for this question. It may have been stopped.</p>
       ${question ? `<div class="answer-actions"><div class="action-group"><button class="btn small" type="button" data-act="retry">${ICON.retry} Ask again</button></div></div>` : ""}`;
  return `
    <section class="${classes}" data-turn="${index}" id="turn-${index}" aria-label="Question ${index + 1}">
      <div class="answer-kicker"><span>${escapeHtml(status.kicker)}</span></div>
      <h2 class="turn-question">${escapeHtml(question?.content || "Earlier answer")}</h2>
      ${body}
    </section>`;
}

function pendingHtml(pending) {
  const steps = PROGRESS.map(
    (step, i) => `<li class="${i < pending.step ? "done" : i === pending.step ? "current" : ""}"><i aria-hidden="true"></i><span>${step.label}</span></li>`
  ).join("");
  return `
    <section class="turn pending" id="turn-pending" aria-label="Answering">
      <div class="answer-kicker">${loader("small")}<span>Working on it</span></div>
      <h2 class="turn-question">${escapeHtml(pending.question)}</h2>
      <div class="progress-strip"><ol>${steps}</ol><span class="elapsed" id="pendingElapsed">0s</span></div>
      <p class="status-line" id="pendingMessage" role="status">${escapeHtml(pending.message)}</p>
      <div class="answer-skeleton" aria-hidden="true"><i></i><i></i><i></i><i></i></div>
    </section>`;
}

function emptyState() {
  const docs = state.documents.filter((doc) => doc.status === "indexed" && doc.included !== false);
  const prompts = [];
  if (state.demo) prompts.push(...SAMPLE_QUESTIONS.map((q) => ({ q, label: q })));
  else {
    docs.slice(0, 3).forEach((doc) => {
      const q = `What are the key points in “${docTitle(doc.name)}”?`;
      prompts.push({ q, label: q });
    });
    state.conversations.slice(0, 2).forEach((thread) => {
      if (thread.title && !prompts.some((p) => p.q === thread.title)) prompts.push({ q: thread.title, label: thread.title, recent: true });
    });
  }
  const noDocs = !docs.length && !state.demo;
  const heading = state.demo
    ? "Ask a question about machine learning."
    : noDocs
      ? "Start by adding a document."
      : "Ask across the documents you have indexed.";
  const text = state.demo
    ? "This demo has a few machine-learning documents loaded: notes on RNNs and on information retrieval, an attention-mechanism deck, and a short course on autoencoders. Every answer cites the passages it came from, and the system says so when the documents do not answer."
    : noDocs
      ? "Needle answers only from what you upload, and every answer shows the passage it came from."
      : "Every answer shows the passages it came from. When your documents don't cover a question, Needle says so.";
  const action = noDocs
    ? isOwner()
      ? `<button type="button" class="btn primary" data-empty-upload>${ICON.upload} Upload your first document</button>`
      : ""
    : `<div class="sample-questions">${prompts
        .slice(0, 4)
        .map(
          (p) => `<button type="button" class="sample-question" data-q="${escapeHtml(p.q)}">${p.recent ? `<span class="sample-kind">Ask again</span>` : ""}${escapeHtml(p.label)}</button>`
        )
        .join("")}</div>`;
  return `
    <div class="home-empty">
      <div class="logo-build">${LOGO_SVG}</div>
      <div class="answer-kicker"><span>Grounded answers</span></div>
      <h2>${heading}</h2>
      <p>${text}</p>
      ${action}
    </div>`;
}

let turnObserver = null;
let manualTurnUntil = 0;

function renderConversation() {
  const root = $("#conversationInner");
  turnObserver?.disconnect();
  const turns = conversationTurns(state.messages);
  const pending = state.pending;
  if (!turns.length && !pending) {
    root.innerHTML = emptyState();
    $$(".sample-question", root).forEach((button) => listen(button, "click", () => sendQuestion(button.dataset.q)));
    listen($("[data-empty-upload]", root), "click", () => {
      showPage("knowledge");
      chooseFile("upload");
    });
    renderInspector(null);
    setTitle("Ask");
    return;
  }
  if (state.activeTurn == null || state.activeTurn >= turns.length) state.activeTurn = turns.length - 1;
  const head = state.conversationId
    ? `<div class="thread-head"><span class="section-eyebrow">Conversation</span><strong>${escapeHtml(activeConversationTitle())}</strong>
        <button type="button" class="thread-more head-more" aria-label="Conversation options" aria-haspopup="menu" aria-expanded="false">${ICON.more}</button></div>`
    : "";
  root.innerHTML = head + turns.map(turnHtml).join("") + (pending ? pendingHtml(pending) : "");
  const headMore = $(".head-more", root);
  listen(headMore, "click", (event) => {
    event.stopPropagation();
    threadMenu(headMore, state.conversationId);
  });
  $$(".turn[data-turn]", root).forEach((section) => wireTurn(section, turns[Number(section.dataset.turn)], Number(section.dataset.turn)));
  if (pending) renderWorkingInspector();
  else if (turns.length) renderInspector(turns[state.activeTurn]?.answer || null, state.activeTurn);
  setTitle(activeConversationTitle());
  if (state.revealTurn != null) {
    const reveal = state.revealTurn;
    setTimeout(() => {
      if (state.revealTurn === reveal) state.revealTurn = null;
    }, 1600);
  }
  if (!pending && turns.length > 1 && "IntersectionObserver" in window) {
    // The evidence panel follows the turn being read.
    turnObserver = new IntersectionObserver(
      (entries) => {
        if (Date.now() < manualTurnUntil) return;
        const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) return;
        const index = Number(visible.target.dataset.turn);
        if (index !== state.activeTurn) setActiveTurn(index);
      },
      { root: phone() ? null : $("#conversation"), threshold: [0.35, 0.6] }
    );
    $$(".turn[data-turn]", root).forEach((section) => turnObserver.observe(section));
  }
}

function setActiveTurn(index, { manual = false } = {}) {
  if (manual) manualTurnUntil = Date.now() + 1200;
  if (index === state.activeTurn) return;
  state.activeTurn = index;
  $$(".turn[data-turn]").forEach((section) => section.classList.toggle("active", Number(section.dataset.turn) === index));
  const turns = conversationTurns(state.messages);
  renderInspector(turns[index]?.answer || null, index);
}

function wireTurn(section, turn, index) {
  const answer = turn?.answer;
  const question = turn?.question?.content || "";
  $$(".citation", section).forEach((button) => {
    const n = button.dataset.source;
    button.addEventListener("mouseenter", () => hoverSource(index, n, true));
    button.addEventListener("mouseleave", () => hoverSource(index, n, false));
    button.addEventListener("focus", () => hoverSource(index, n, true));
    button.addEventListener("blur", () => hoverSource(index, n, false));
    listen(button, "click", () => openSource(index, n, button));
  });
  $$("[data-act]", section).forEach((button) => {
    const act = button.dataset.act;
    listen(button, "click", async () => {
      if (act === "copy") return copyText(plainText(answer?.content || ""), "Answer copied");
      if (act === "copy-sources") return copyText(withSources(answer), "Answer and sources copied");
      if (act === "retry") return sendQuestion(question);
      if (act === "edit") {
        const input = $("#askInput");
        input.value = question;
        input.dispatchEvent(new Event("input"));
        input.focus();
        input.setSelectionRange(question.length, question.length);
        return;
      }
      if (act === "evidence") {
        setActiveTurn(index, { manual: true });
        if (drawerMode()) setInspector(true);
        return;
      }
      if (act === "helpful" || act === "unhelpful") return rate(answer, act, section);
    });
  });
  $$("[data-reason]", section).forEach((button) =>
    listen(button, "click", async () => {
      await api.feedback(answer.id, "unhelpful", button.dataset.reason);
      const box = $(".feedback-reasons", section);
      box.innerHTML = `<span>Thanks — this helps tune the answers.</span>`;
      setTimeout(() => {
        box.hidden = true;
      }, 2400);
    })
  );
}

function withSources(answer) {
  const sources = answer?.sources || [];
  const list = sources
    .map((source, i) => {
      const where = [source.header_context, source.page_number ? `page ${source.page_number}` : ""].filter(Boolean).join(", ");
      return `[${i + 1}] ${source.document_name || "Source"}${where ? ` — ${where}` : ""}`;
    })
    .join("\n");
  return `${plainText(answer?.content || "")}${list ? `\n\nSources\n${list}` : ""}`;
}

async function copyText(text, done) {
  if (!text.trim()) throw new Error("There is nothing to copy.");
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
  notify(done, { type: "success" });
}

async function rate(answer, rating, section) {
  if (!answer?.id) {
    notify("This answer hasn't been saved yet. Try again in a moment.");
    return;
  }
  const previous = answer.rating || null;
  const next = previous === rating ? "none" : rating;
  await api.feedback(answer.id, next);
  answer.rating = next === "none" ? null : next;
  const up = $('[data-act="helpful"]', section);
  const down = $('[data-act="unhelpful"]', section);
  up.classList.toggle("active", answer.rating === "helpful");
  up.setAttribute("aria-pressed", String(answer.rating === "helpful"));
  down.classList.toggle("active", answer.rating === "unhelpful");
  down.setAttribute("aria-pressed", String(answer.rating === "unhelpful"));
  $(".feedback-reasons", section).hidden = answer.rating !== "unhelpful";
  if (next === "none") return;
  notify(next === "helpful" ? "Marked as helpful" : "Marked as not helpful", {
    action: {
      label: "Undo",
      onClick: async () => {
        await api.feedback(answer.id, previous || "none");
        answer.rating = previous;
        renderConversation();
      },
    },
  });
}

// ---------------------------------------------------------------------------------------
// Inspector: trace and sources for the active turn
// ---------------------------------------------------------------------------------------

function traceNodes(states) {
  const labels = ["Query", "Search", "Rerank", "Verify"];
  return `<div class="trace-flow">${labels
    .map((label, i) => {
      const node = states[i] || "";
      const line = i ? `<i class="trace-line ${node === "done" || node === "active" ? "done" : ""}"></i>` : "";
      return `${line}<div class="trace-node ${node}"><i class="node-dot"></i><span>${label}</span></div>`;
    })
    .join("")}</div>`;
}

function workingStates(step) {
  if (step <= 0) return ["done", "active", "", ""];
  if (step === 1) return ["done", "done", "active", ""];
  return ["done", "done", "done", "active"];
}

function renderWorkingInspector() {
  const pending = state.pending;
  $("#inspectorBody").innerHTML = `
    <section class="trace working">
      <div class="trace-head"><span>Retrieval trace</span><span class="trace-time" id="traceElapsed">0s</span></div>
      <div id="traceNodes">${traceNodes(workingStates(pending?.step || 0))}</div>
      <p class="trace-note" id="traceNote">${escapeHtml(pending?.message || "")}</p>
    </section>
    <div class="source-title"><strong>Supporting sources</strong><span>Finding…</span></div>
    <div class="source-list" aria-hidden="true">${'<div class="source sk-source"><i></i><i></i><i></i></div>'.repeat(3)}</div>`;
}

function renderInspector(message, turnIndex = state.activeTurn) {
  const body = $("#inspectorBody");
  const sources = message?.sources || [];
  const trace = message?.trace || {};
  const showTrace = state.settings?.show_traces !== false;
  if (!message) {
    body.innerHTML = `<p class="empty-note">Ask a question to see the evidence for the answer.</p>`;
    return;
  }
  if (!showTrace && !sources.length) {
    body.innerHTML = `<p class="empty-note">No sources were kept for this answer. Retrieval traces are hidden in Settings.</p>`;
    return;
  }
  const validation = message.validation || {};
  const states = [
    trace.original_query || trace.retrieval_query ? "done" : "",
    trace.candidates != null ? "done" : "",
    trace.rerank_mode ? "done" : "",
    validation.passed ? "done" : "",
  ];
  const traceBlock = showTrace
    ? `<section class="trace">
      <div class="trace-head"><span>Retrieval trace</span><span class="trace-time">${trace.latencies_ms?.total ? `${escapeHtml(seconds(trace.latencies_ms.total))} total` : `${trace.candidates ?? 0} candidates`}</span></div>
      ${traceNodes(states)}
      <div class="trace-stats">
        <div><strong>${trace.candidates ?? 0} → ${trace.kept ?? sources.length}</strong><span>passages kept</span></div>
        <div><strong>${Number(trace.similarity_threshold ?? state.settings?.similarity_threshold ?? 0).toFixed(2)}</strong><span>min similarity</span></div>
        <div><strong>${escapeHtml(String(trace.version_id || state.index?.version_id || "").slice(0, 8) || "—")}</strong><span>index version</span></div>
      </div>
      <details><summary>Raw trace</summary><pre>${escapeHtml(JSON.stringify(trace, null, 2))}</pre></details>
    </section>`
    : "";
  const cited = citedNumbers(message.content);
  const threshold = Number(state.index?.jev_relevance_threshold ?? 0.2);
  const card = (source, i) => {
    const n = i + 1;
    const score = source.similarity_score;
    const weak = source.relation === "related" || (score != null && score < threshold);
    const full = String(source.text || source.matched_passage || "");
    const match = String(source.matched_passage || "").trim();
    let detail = escapeHtml(truncate(full, 1400));
    const probe = match.slice(0, 80);
    const at = probe ? full.indexOf(probe) : -1;
    if (at >= 0) {
      const end = Math.min(full.length, at + match.length);
      const from = Math.max(0, at - 500);
      detail = `${from ? "…" : ""}${escapeHtml(full.slice(from, at))}<mark class="sweep">${escapeHtml(full.slice(at, end))}</mark>${escapeHtml(full.slice(end, end + 500))}${end + 500 < full.length ? "…" : ""}`;
    }
    return `
      <article class="source ${weak ? "weak" : ""} ${cited.has(n) ? "cited" : ""}" data-id="${n}" data-doc="${escapeHtml(source.document_id || "")}" data-parent="${escapeHtml(source.parent_id || "")}" data-probe="${escapeHtml(match.slice(0, 80))}">
        <i class="fold" aria-hidden="true"></i>
        <button type="button" class="source-main" aria-expanded="false" aria-controls="source-detail-${turnIndex}-${n}">
          <div class="source-top"><span class="source-num">${n}</span><span class="source-type"><span class="file-badge">${fileKind(source.document_name)}</span>${escapeHtml(source.document_name || "Source")}</span><span class="score">${score != null ? `${Math.round(score * 100)}%` : "—"}</span></div>
          <h3>${escapeHtml(source.header_context || `Page ${source.page_number || "—"}`)}</h3>
          <p class="source-snippet">${escapeHtml(truncate((source.matched_passage || source.text || "").replace(/\s+/g, " "), 220))}</p>
        </button>
        <div class="source-detail" id="source-detail-${turnIndex}-${n}" hidden>
          <p class="source-full">${detail}</p>
          <div class="source-foot"><span>${source.page_number ? `Page ${source.page_number}` : ""}${weak ? " · weak match" : ""}</span>${source.document_id ? `<button type="button" class="link-button" data-open-doc>Open in document →</button>` : ""}</div>
        </div>
      </article>`;
  };
  const supporting = sources.map((source, i) => ({ source, i })).filter(({ source }) => source.relation !== "related");
  const related = sources.map((source, i) => ({ source, i })).filter(({ source }) => source.relation === "related");
  const list = (items) => items.map(({ source, i }) => card(source, i)).join("");
  body.innerHTML = `
    ${traceBlock}
    <div class="source-title"><strong>${supporting.length ? "Supporting sources" : "Related passages"}</strong><span>${sources.length} match${sources.length === 1 ? "" : "es"} · relevance</span></div>
    <div class="source-list">${sources.length ? list(supporting.length ? supporting : related) : `<p class="empty-note">No citation was kept for this answer.</p>`}</div>
    ${supporting.length && related.length ? `<div class="source-title related-title"><strong>Related, not used</strong><span>${related.length}</span></div><div class="source-list">${list(related)}</div>` : ""}`;
  $$(".source", body).forEach((article) => {
    listen($(".source-main", article), "click", () => toggleSource(article, turnIndex));
    listen($("[data-open-doc]", article), "click", () =>
      // Older answers were saved without passage ids; their text finds the passage instead.
      openDocument(article.dataset.doc, article.dataset.parent || null, article.dataset.parent ? null : article.dataset.probe || null)
    );
  });
}

function toggleSource(article, turnIndex, open) {
  const detail = $(".source-detail", article);
  const next = open ?? detail.hidden;
  detail.hidden = !next;
  $(".source-main", article).setAttribute("aria-expanded", String(next));
  article.classList.toggle("expanded", next);
  const section = $(`.turn[data-turn="${turnIndex}"]`);
  $$(".citation", section || document).forEach((cite) => cite.classList.toggle("active", next && cite.dataset.source === article.dataset.id));
  if (next) {
    const mark = $("mark.sweep", detail);
    if (mark) {
      mark.classList.remove("go");
      void mark.offsetWidth;
      mark.classList.add("go");
    }
  }
}

function hoverSource(turnIndex, n, on) {
  if (turnIndex !== state.activeTurn) return;
  $(`#inspectorBody .source[data-id="${n}"]`)?.classList.toggle("hover", on);
}

function openSource(turnIndex, n, citation) {
  setActiveTurn(turnIndex, { manual: true });
  if (drawerMode()) setInspector(true);
  const article = $(`#inspectorBody .source[data-id="${n}"]`);
  if (!article) {
    notify(`This answer has no source numbered ${n}.`);
    return;
  }
  $$("#inspectorBody .source.expanded").forEach((other) => other !== article && toggleSource(other, turnIndex, false));
  toggleSource(article, turnIndex, true);
  citation?.classList.add("active");
  article.scrollIntoView({ block: "nearest", behavior: reducedMotion() ? "auto" : "smooth" });
  article.classList.remove("pulse");
  void article.offsetWidth;
  article.classList.add("pulse");
}

// ---------------------------------------------------------------------------------------
// Asking
// ---------------------------------------------------------------------------------------

async function openConversation(id) {
  if (state.busy) {
    notify("Still answering. Stop the current answer first.");
    return;
  }
  const data = await api.conversation(id);
  state.conversationId = id;
  state.messages = data.messages || [];
  state.activeTurn = null;
  state.revealTurn = null;
  await showPage("ask");
  renderThreads();
  renderConversation();
  const turns = $$(".turn[data-turn]");
  turns[turns.length - 1]?.scrollIntoView({ block: "start" });
}

function startNewConversation({ focus = true } = {}) {
  if (state.busy) {
    notify("Still answering. Stop the current answer first.");
    return;
  }
  state.conversationId = null;
  state.messages = [];
  state.activeTurn = null;
  if (state.page !== "ask") showPage("ask");
  renderThreads();
  renderConversation();
  if (focus) $("#askInput").focus();
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
    if (event.type === "status") onStatus(event);
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
}

function setComposerBusy(busy) {
  const form = $("#askForm");
  form.classList.toggle("busy", busy);
  const send = $("#sendButton");
  send.setAttribute("aria-label", busy ? "Stop answering" : "Send question");
  send.dataset.tip = busy ? "Stop" : "";
  $("#conversationInner").setAttribute("aria-busy", busy ? "true" : "false");
}

function updatePending(event) {
  const pending = state.pending;
  if (!pending) return;
  if (event?.stage) pending.step = Math.max(pending.step, stepOf(event.stage));
  if (event?.message) pending.message = event.message;
  $$("#turn-pending .progress-strip li").forEach((li, i) => {
    li.classList.toggle("done", i < pending.step);
    li.classList.toggle("current", i === pending.step);
  });
  const message = $("#pendingMessage");
  if (message) message.textContent = pending.message;
  const nodes = $("#traceNodes");
  if (nodes) nodes.innerHTML = traceNodes(workingStates(pending.step));
  const note = $("#traceNote");
  if (note) note.textContent = pending.message;
}

function tickElapsed() {
  if (!state.pending) return;
  const text = `${Math.floor((Date.now() - state.pending.startedAt) / 1000)}s`;
  const a = $("#pendingElapsed");
  const b = $("#traceElapsed");
  if (a) a.textContent = text;
  if (b) b.textContent = text;
}

function scrollToTurn(selector) {
  const node = typeof selector === "string" ? $(selector) : selector;
  node?.scrollIntoView({ block: "start", behavior: reducedMotion() ? "auto" : "smooth" });
}

async function sendQuestion(query) {
  query = String(query || "").trim();
  if (!query) return;
  if (state.busy) {
    notify("Still answering. Stop the current answer first.");
    return;
  }
  if (state.page !== "ask") await showPage("ask");
  state.busy = true;
  const controller = new AbortController();
  state.pending = { question: query, step: 0, message: "Reading your question…", startedAt: Date.now(), controller };
  closeContextMenu();
  setComposerBusy(true);
  renderConversation();
  scrollToTurn("#turn-pending");
  const timer = setInterval(tickElapsed, 500);
  const prior = state.messages.slice();
  try {
    const response = await api.chat({ query, document_id: state.scopeId || null, conversation_id: state.conversationId }, controller.signal);
    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(errorMessage(err, "The question could not be answered. Try again."));
    }
    const result = await readChatStream(response, updatePending);
    try {
      await reloadConversation();
    } catch (err) {
      state.messages = prior.concat([
        { role: "user", content: query },
        { id: result.messageId, role: "assistant", content: result.answer, sources: result.sources, validation: result.validation, trace: result.trace },
      ]);
      notify(err.message, { type: "error" });
    }
    state.pending = null;
    const turns = conversationTurns(state.messages);
    state.revealTurn = turns.length - 1;
    state.activeTurn = turns.length - 1;
    await refreshShell().catch(() => {});
    renderConversation();
    scrollToTurn(`#turn-${state.revealTurn}`);
  } catch (err) {
    state.pending = null;
    const stopped = err.name === "AbortError";
    if (state.conversationId) {
      try {
        await reloadConversation();
      } catch {
        state.messages = prior;
      }
    } else state.messages = prior;
    await refreshShell().catch(() => {});
    renderConversation();
    if (stopped) notify("Stopped. That question wasn't answered.", { action: { label: "Ask again", onClick: () => sendQuestion(query) } });
    else notify(err.message, { type: "error", action: { label: "Try again", onClick: () => sendQuestion(query) } });
  } finally {
    clearInterval(timer);
    state.busy = false;
    state.pending = null;
    setComposerBusy(false);
  }
}

// ---------------------------------------------------------------------------------------
// Composer: scope, voice
// ---------------------------------------------------------------------------------------

function updateScopeLabel() {
  const doc = state.documents.find((item) => item.id === state.scopeId);
  $("#contextLabel").textContent = doc ? truncate(docTitle(doc.name), 28) : "All documents";
  $("#addContext").classList.toggle("scoped", Boolean(doc));
  $("#addContext").title = doc ? `Answers use only ${doc.name}` : "Answers use every included document";
}

function closeContextMenu() {
  const menu = $("#contextMenu");
  if (menu.hidden) return false;
  menu.hidden = true;
  $("#addContext").setAttribute("aria-expanded", "false");
  return true;
}

function renderContextMenu() {
  const menu = $("#contextMenu");
  const docs = state.documents.filter((doc) => doc.status === "indexed" && doc.included !== false);
  const option = (id, label, sub = "") =>
    `<button type="button" role="option" data-scope="${escapeHtml(id)}" aria-selected="${(state.scopeId || "") === id}"><span>${escapeHtml(label)}</span>${sub ? `<small>${escapeHtml(sub)}</small>` : ""}</button>`;
  menu.innerHTML = `
    <div class="menu-heading">Search in</div>
    ${docs.length > 6 ? `<input class="menu-search" id="scopeSearch" placeholder="Find a document" aria-label="Find a document" />` : ""}
    <div class="menu-options">${option("", "All documents", `${docs.length} included`)}${docs.map((doc) => option(doc.id, docTitle(doc.name), doc.name)).join("")}</div>`;
  $$("[data-scope]", menu).forEach((button) =>
    listen(button, "click", () => {
      state.scopeId = button.dataset.scope || null;
      updateScopeLabel();
      closeContextMenu();
      $("#askInput").focus();
    })
  );
  const search = $("#scopeSearch");
  if (search) {
    search.addEventListener("input", () => {
      const q = search.value.trim().toLowerCase();
      $$("[data-scope]", menu).forEach((button) => {
        button.hidden = Boolean(button.dataset.scope) && !button.textContent.toLowerCase().includes(q);
      });
    });
    search.focus();
  } else $(`[aria-selected="true"]`, menu)?.focus();
}

let recorder = null;
let recordTimer = null;
let meter = null;

function stopRecording() {
  if (recorder && recorder.state !== "inactive") recorder.stop();
}

function voiceIdle(label = "Voice") {
  const button = $("#voiceButton");
  button.classList.remove("recording", "busy");
  button.style.removeProperty("--level");
  button.setAttribute("aria-pressed", "false");
  $("#voiceLabel").textContent = label;
  $("#composerHint").textContent = "Enter to send · Shift+Enter for a new line";
  $("#askInput").placeholder = "Ask across your knowledge base…";
}

function startMeter(stream, startedAt) {
  const button = $("#voiceButton");
  let context = null;
  let frame = 0;
  try {
    context = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = context.createAnalyser();
    analyser.fftSize = 512;
    context.createMediaStreamSource(stream).connect(analyser);
    const data = new Uint8Array(analyser.fftSize);
    const loop = () => {
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (const v of data) sum += ((v - 128) / 128) ** 2;
      button.style.setProperty("--level", Math.min(1, Math.sqrt(sum / data.length) * 4).toFixed(3));
      const secs = Math.floor((Date.now() - startedAt) / 1000);
      $("#voiceLabel").textContent = `0:${String(secs).padStart(2, "0")} · Stop`;
      frame = requestAnimationFrame(loop);
    };
    loop();
  } catch {
    // The level ring is decoration; recording works without it.
  }
  return () => {
    cancelAnimationFrame(frame);
    context?.close().catch(() => {});
  };
}

async function toggleVoice() {
  if (recorder) {
    stopRecording();
    return;
  }
  if (!window.MediaRecorder || !navigator.mediaDevices?.getUserMedia) {
    notify("Voice input is not available in this browser.", { type: "error" });
    return;
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    notify("Microphone access was blocked. Allow it in the browser's site settings to use voice.", { type: "error" });
    return;
  }
  const chunks = [];
  const startedAt = Date.now();
  const active = new MediaRecorder(stream, { audioBitsPerSecond: 24000 });
  recorder = active;
  const button = $("#voiceButton");
  active.ondataavailable = (event) => {
    if (event.data.size) chunks.push(event.data);
  };
  active.onstop = async () => {
    clearTimeout(recordTimer);
    meter?.();
    meter = null;
    stream.getTracks().forEach((track) => track.stop());
    if (recorder === active) recorder = null;
    const secs = (Date.now() - startedAt) / 1000;
    if (!chunks.length || secs < 0.5) {
      voiceIdle();
      notify("No speech was heard.");
      return;
    }
    button.classList.remove("recording");
    button.classList.add("busy");
    button.disabled = true;
    $("#voiceLabel").textContent = "Transcribing…";
    $("#askInput").placeholder = "Transcribing your question…";
    try {
      const { text } = await api.transcribe(new Blob(chunks, { type: active.mimeType }), secs);
      if (!text) {
        notify("No speech was heard.");
        return;
      }
      const input = $("#askInput");
      input.value = text;
      input.dispatchEvent(new Event("input"));
      input.focus();
    } catch (err) {
      notify(err.message || "Voice input failed.", { type: "error" });
    } finally {
      button.disabled = false;
      voiceIdle();
    }
  };
  active.start();
  button.classList.add("recording");
  button.setAttribute("aria-pressed", "true");
  $("#composerHint").textContent = `Listening… press Stop when you're done (up to ${state.voiceSeconds} seconds)`;
  meter = startMeter(stream, startedAt);
  // The server also refuses long clips; stopping here keeps the cost of one recording bounded.
  recordTimer = setTimeout(stopRecording, state.voiceSeconds * 1000);
}

// ---------------------------------------------------------------------------------------
// Uploads: queue with per-file progress
// ---------------------------------------------------------------------------------------

const UPLOAD_STEPS = ["Upload", "Queue", "Index", "Ready"];
const uploadStep = { waiting: -1, uploading: 0, queued: 1, processing: 2, indexed: 4, duplicate: 4, failed: -2 };

function chooseFile(mode = "upload", { replaceId = null, forQuestion = null } = {}) {
  if (!isOwner()) return;
  const input = $("#fileInput");
  input.dataset.replace = mode === "replace" && replaceId ? replaceId : "";
  input.dataset.question = forQuestion || "";
  input.multiple = mode !== "replace";
  input.value = "";
  input.click();
}

function addFiles(fileList, { replaceId = null, forQuestion = null } = {}) {
  if (!isOwner()) return;
  const files = [...(fileList || [])];
  if (!files.length) return;
  const accepted = files.filter((file) => ACCEPTED.test(file.name));
  const skipped = files.filter((file) => !ACCEPTED.test(file.name));
  if (skipped.length) {
    notify(`Skipped ${skipped.map((file) => file.name).join(", ")}: only PDF, Word, PowerPoint, Excel, CSV, Markdown, and text files can be indexed.`, { type: "error" });
  }
  accepted.forEach((file, i) =>
    state.uploads.push({
      key: `${Date.now()}-${i}-${file.name}`,
      file,
      name: file.name,
      size: file.size,
      status: "waiting",
      error: "",
      replaceId: i === 0 ? replaceId : null,
      forQuestion,
    })
  );
  if (!accepted.length) return;
  if (state.page !== "knowledge") {
    notify(`${accepted.length} file${accepted.length === 1 ? "" : "s"} added to the upload queue`, {
      action: { label: "View", onClick: () => showPage("knowledge") },
    });
  }
  paintUploads();
  processUploads();
}

async function processUploads() {
  if (state.uploading) return;
  state.uploading = true;
  try {
    for (;;) {
      const item = state.uploads.find((upload) => upload.status === "waiting");
      if (!item) break;
      await runUpload(item);
    }
  } finally {
    state.uploading = false;
  }
}

async function runUpload(item) {
  item.status = "uploading";
  paintUploads();
  try {
    const queued = await api.upload(item.file);
    if (queued.status === "duplicate") {
      item.status = "duplicate";
      item.docId = queued.id;
      paintUploads();
      notify(`${item.name} is already indexed`, { action: { label: "Open", onClick: () => openDocument(queued.id) } });
      return;
    }
    item.status = "queued";
    paintUploads();
    await refreshShell();
    if (state.page === "knowledge") paintDocuments();
    const deadline = Date.now() + 30 * 60 * 1000;
    let delay = 800;
    let job = null;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, delay));
      delay = Math.min(delay * 1.4, 4000);
      job = await api.job(queued.job_id);
      if (job.status !== item.status && (job.status === "queued" || job.status === "processing")) {
        item.status = job.status;
        paintUploads();
      }
      if (job.status === "completed" || job.status === "failed") break;
    }
    if (!job || job.status === "queued" || job.status === "processing") {
      throw new Error("Still indexing. It will appear in the table when it finishes.");
    }
    if (job.status === "failed") throw new Error(job.error || "The file could not be indexed.");
    item.status = "indexed";
    item.docId = job.document_id;
    if (item.replaceId && item.replaceId !== job.document_id) {
      try {
        await api.deleteDocument(item.replaceId);
      } catch (err) {
        notify(`${item.name} was indexed, but the previous version could not be removed. ${err.message}`, { type: "error" });
      }
    }
    paintUploads();
    await refreshShell();
    if (state.page === "knowledge") await renderKnowledge();
    if (item.replaceId) {
      state.activeDocumentId = job.document_id;
      state.focusPassage = null;
      showPage("document");
    }
    notify(`${item.name} is ready to search`, {
      type: "success",
      action: item.forQuestion
        ? { label: "Ask again", onClick: () => sendQuestion(item.forQuestion) }
        : { label: "Open", onClick: () => openDocument(job.document_id) },
    });
  } catch (err) {
    item.status = "failed";
    item.error = err.message;
    paintUploads();
    notify(`${item.name}: ${err.message}`, { type: "error" });
  }
}

function paintUploads() {
  const panel = $("#uploadPanel");
  if (!panel) return;
  const items = state.uploads;
  panel.hidden = !items.length;
  if (!items.length) return;
  const active = items.filter((item) => !["indexed", "duplicate", "failed"].includes(item.status)).length;
  panel.innerHTML = `
    <div class="panel-head"><div><h2>Uploads</h2><p>${active ? `${active} in progress` : "All done"}</p></div>
      <button type="button" class="btn small" id="clearUploads" ${active === items.length ? "disabled" : ""}>Clear finished</button></div>
    <div class="upload-list">${items
      .map((item) => {
        const step = uploadStep[item.status];
        const label =
          item.status === "failed"
            ? item.error
            : item.status === "duplicate"
              ? "Already indexed"
              : item.status === "indexed"
                ? "Ready to search"
                : item.status === "waiting"
                  ? "Waiting"
                  : item.status === "uploading"
                    ? "Uploading…"
                    : item.status === "queued"
                      ? "Queued for indexing"
                      : "Parsing, chunking, and embedding…";
        const steps = UPLOAD_STEPS.map((name, i) => `<li class="${step > i || step === 4 ? "done" : step === i ? "current" : ""}"><i></i><span>${name}</span></li>`).join("");
        return `
          <div class="upload-row ${item.status}" data-key="${escapeHtml(item.key)}">
            <div class="doc-icon">${fileKind(item.name)}</div>
            <div class="upload-meta"><strong>${escapeHtml(item.name)}</strong><span>${bytes(item.size)} · ${escapeHtml(label)}</span></div>
            <ol class="upload-steps" aria-label="Progress">${steps}</ol>
            <div class="upload-actions">
              ${item.status === "failed" ? `<button type="button" class="btn small" data-upload-retry>Try again</button>` : ""}
              ${item.docId && item.status !== "failed" ? `<button type="button" class="btn small" data-upload-open>Open</button>` : ""}
              ${["indexed", "duplicate", "failed"].includes(item.status) ? `<button type="button" class="row-action" data-upload-dismiss aria-label="Dismiss ${escapeHtml(item.name)}">×</button>` : ""}
            </div>
          </div>`;
      })
      .join("")}</div>`;
  listen($("#clearUploads"), "click", () => {
    state.uploads = state.uploads.filter((item) => !["indexed", "duplicate", "failed"].includes(item.status));
    paintUploads();
  });
  $$(".upload-row", panel).forEach((row) => {
    const item = state.uploads.find((upload) => upload.key === row.dataset.key);
    listen($("[data-upload-retry]", row), "click", () => {
      item.status = "waiting";
      item.error = "";
      paintUploads();
      processUploads();
    });
    listen($("[data-upload-open]", row), "click", () => openDocument(item.docId));
    listen($("[data-upload-dismiss]", row), "click", () => {
      state.uploads = state.uploads.filter((upload) => upload !== item);
      paintUploads();
    });
  });
}

// Drop files anywhere (owners only).
let dragDepth = 0;
const hasFiles = (event) => [...(event.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (event) => {
  if (!hasFiles(event) || !isOwner() || !$("#modal").hidden) return;
  event.preventDefault();
  dragDepth += 1;
  $("#dropOverlay").hidden = false;
});
window.addEventListener("dragover", (event) => {
  if (hasFiles(event) && isOwner()) event.preventDefault();
});
window.addEventListener("dragleave", (event) => {
  if (!hasFiles(event)) return;
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) $("#dropOverlay").hidden = true;
});
window.addEventListener("drop", (event) => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  dragDepth = 0;
  $("#dropOverlay").hidden = true;
  if (!isOwner()) return;
  addFiles(event.dataTransfer.files);
  if (state.page !== "knowledge" && state.page !== "document") showPage("knowledge");
});

// ---------------------------------------------------------------------------------------
// Knowledge base
// ---------------------------------------------------------------------------------------

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
  const excluded = indexed.filter((doc) => doc.included === false).length;
  const first = !$("#page-knowledge .stat-card");
  $("#page-knowledge").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Knowledge operations</div><h1>Knowledge base</h1><p>Curate the source material that can be searched, cited, and used for grounded answers.</p></div>
        <div class="page-actions">
          <button class="btn primary" id="uploadButton" type="button" data-owner>${ICON.upload} Upload files</button>
        </div>
      </header>
      <div class="stat-grid">
        <article class="stat-card accent"><span>Documents</span><div class="stat-value" data-count="${indexed.length}">${indexed.length}</div><small>${recent ? `+${recent} this month` : "None added this month"}${processing.length ? ` · ${processing.length} in progress` : ""}${excluded ? ` · ${excluded} excluded` : ""}</small></article>
        <article class="stat-card"><span>Indexed chunks</span><div class="stat-value" data-count="${chunks}">${chunks.toLocaleString()}</div><small>Across ${collections} collection${collections === 1 ? "" : "s"}</small></article>
        <article class="stat-card"><span>Storage used</span><div class="stat-value">${storage ? bytes(storage) : "—"}</div><small>Original files kept on this machine</small></article>
        <article class="stat-card dark"><span>Index status</span><div class="stat-value">${state.index?.compatible === false ? "Refresh" : "Healthy"}</div><small>${state.index?.compatible === false ? "Embedding model changed" : lastHandoff ? `Last handoff ${escapeHtml(whenInline(lastHandoff))}` : escapeHtml(state.index?.embedding_model || "")}</small></article>
      </div>
      <button type="button" class="dropzone" id="dropzone" data-owner>
        ${ICON.upload}
        <span><strong>Drop files here, or browse</strong><small>PDF, Word, PowerPoint, Excel, CSV, Markdown, or text. Several at once is fine.</small></span>
      </button>
      <section class="panel upload-panel" id="uploadPanel" data-owner hidden></section>
      <section class="panel">
        <div class="panel-head">
          <div><h2>All documents</h2><p>Sources available to answers in this workspace</p></div>
          <div class="panel-tools">
            <label class="filter-field"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg><input id="documentSearch" type="search" value="${escapeHtml(state.docQuery)}" placeholder="Search documents" aria-label="Search documents" /></label>
            <div class="segmented" id="docFilter" role="group" aria-label="Filter by status">
              <button type="button" class="${state.docFilter === "all" ? "active" : ""}" data-filter="all" aria-pressed="${state.docFilter === "all"}">All</button>
              <button type="button" class="${state.docFilter === "indexed" ? "active" : ""}" data-filter="indexed" aria-pressed="${state.docFilter === "indexed"}">Indexed</button>
              <button type="button" class="${state.docFilter === "processing" ? "active" : ""}" data-filter="processing" aria-pressed="${state.docFilter === "processing"}">In progress</button>
            </div>
          </div>
        </div>
        <div class="table-scroll">
          <table class="data-table docs-table"><thead><tr>
            ${sortHeader("name", "Document")}<th scope="col">Collection</th>${sortHeader("chunk_count", "Chunks")}${sortHeader("uploaded_at", "Added")}<th scope="col">Status</th><th scope="col"><span class="sr-only">Actions</span></th>
          </tr></thead>
          <tbody id="docRows"></tbody></table>
        </div>
      </section>
    </div>`;
  listen($("#uploadButton"), "click", () => chooseFile("upload"));
  listen($("#dropzone"), "click", () => chooseFile("upload"));
  listen($("#documentSearch"), "input", (event) => {
    state.docQuery = event.target.value;
    paintDocuments();
  });
  $$("#docFilter button").forEach((button) =>
    listen(button, "click", () => {
      state.docFilter = button.dataset.filter;
      $$("#docFilter button").forEach((item) => {
        item.classList.toggle("active", item === button);
        item.setAttribute("aria-pressed", String(item === button));
      });
      paintDocuments();
    })
  );
  initSegmented($("#docFilter"));
  $$("[data-sort]", $("#page-knowledge")).forEach((button) =>
    listen(button, "click", () => {
      const key = button.dataset.sort;
      state.docSort = { key, dir: state.docSort.key === key && state.docSort.dir === "asc" ? "desc" : key === "name" ? "asc" : state.docSort.key === key ? "asc" : "desc" };
      renderKnowledge();
    })
  );
  paintUploads();
  paintDocuments();
  if (first) countUp($("#page-knowledge"));
  setTitle("Knowledge base");
}

function sortHeader(key, label) {
  const on = state.docSort.key === key;
  const dir = on ? state.docSort.dir : "none";
  return `<th scope="col" aria-sort="${on ? (dir === "asc" ? "ascending" : "descending") : "none"}"><button type="button" class="sort-button ${on ? "on" : ""}" data-sort="${key}">${label}<span aria-hidden="true">${on ? (dir === "asc" ? "↑" : "↓") : ""}</span></button></th>`;
}

function filteredDocuments() {
  const q = state.docQuery.trim().toLowerCase();
  const { key, dir } = state.docSort;
  const value = (doc) => (key === "name" ? String(doc.name || "").toLowerCase() : key === "chunk_count" ? Number(doc.chunk_count) || 0 : new Date(doc.uploaded_at || 0).getTime());
  return state.documents
    .filter((doc) => {
      const statusOk = state.docFilter === "all" || (state.docFilter === "indexed" ? doc.status === "indexed" : doc.status !== "indexed");
      return statusOk && String(doc.name || "").toLowerCase().includes(q);
    })
    .sort((a, b) => {
      const x = value(a);
      const y = value(b);
      return (x < y ? -1 : x > y ? 1 : 0) * (dir === "asc" ? 1 : -1);
    });
}

function statusLabel(doc) {
  if (doc.status === "indexed") return doc.included === false ? "Excluded" : "Indexed";
  return { queued: "Queued", processing: "Indexing" }[doc.status] || String(doc.status || "Unknown");
}

function paintDocuments() {
  const body = $("#docRows");
  if (!body) return;
  const documents = filteredDocuments();
  if (!documents.length) {
    const note = state.documents.length ? "No documents match this filter." : "No documents yet. Drop files above or use Upload files.";
    body.innerHTML = `<tr><td colspan="6"><p class="empty-note">${note}</p></td></tr>`;
    return;
  }
  body.innerHTML = documents
    .map((doc) => {
      const ready = doc.status === "indexed";
      const pages = doc.max_page ? `${doc.max_page} page${doc.max_page === 1 ? "" : "s"}` : "";
      const meta = [doc.bytes ? bytes(doc.bytes) : "", pages].filter(Boolean).join(" · ");
      return `<tr class="document-row ${ready ? "" : "pending"} ${doc.included === false ? "excluded" : ""}" data-id="${escapeHtml(doc.id)}" ${ready ? 'tabindex="0"' : ""}>
        <td><div class="file-cell"><div class="doc-icon">${fileKind(doc.name)}</div><div><strong>${escapeHtml(doc.name)}</strong><span>${escapeHtml(meta || (ready ? "" : "Waiting for the indexer"))}</span></div></div></td>
        <td>${escapeHtml(doc.collection || "General")}</td>
        <td>${doc.chunk_count ? doc.chunk_count.toLocaleString() : "—"}</td>
        <td>${escapeHtml(doc.uploaded_at ? when(doc.uploaded_at) : "—")}</td>
        <td><span class="status ${ready && doc.included !== false ? "" : "sync"}">${escapeHtml(statusLabel(doc))}</span></td>
        <td class="row-actions">${
          ready
            ? `<button class="row-action" type="button" data-row-toggle data-owner aria-label="${doc.included === false ? "Include" : "Exclude"} ${escapeHtml(doc.name)} ${doc.included === false ? "in" : "from"} answers" data-tip="${doc.included === false ? "Include in answers" : "Exclude from answers"}">${doc.included === false ? ICON.eyeOff : ICON.eye}</button>
               <button class="row-action danger" type="button" data-row-delete data-owner aria-label="Delete ${escapeHtml(doc.name)}" data-tip="Delete">${ICON.trash}</button>
               <span class="row-open" aria-hidden="true">${ICON.chevron}</span>`
            : ""
        }</td>
      </tr>`;
    })
    .join("");
  $$(".document-row", body).forEach((row) => {
    const doc = state.documents.find((item) => item.id === row.dataset.id);
    if (doc?.status !== "indexed") return;
    listen(row, "click", (event) => {
      if (event.target.closest("button")) return;
      openDocument(doc.id);
    });
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && event.target === row) openDocument(doc.id);
    });
    listen($("[data-row-toggle]", row), "click", async () => {
      const next = doc.included === false;
      await api.updateDocument(doc.id, { included: next });
      doc.included = next;
      notify(next ? `${doc.name} is included in answers` : `${doc.name} is excluded from answers`, {
        action: { label: "Undo", onClick: async () => { await api.updateDocument(doc.id, { included: !next }); await renderKnowledge(); } },
      });
      await renderKnowledge();
    });
    listen($("[data-row-delete]", row), "click", () => confirmDelete(doc));
  });
}

// ---------------------------------------------------------------------------------------
// Document detail
// ---------------------------------------------------------------------------------------

async function openDocument(id, passage = null, probe = null) {
  state.activeDocumentId = id;
  state.focusPassage = passage;
  state.focusProbe = probe;
  await showPage("document");
}

// Find a passage by a snippet of its text, loading more of the document if needed.
async function focusByText(probe) {
  const doc = state.doc;
  const text = String(probe || "").replace(/\s+/g, " ").trim().slice(0, 60);
  if (!doc || !text) return null;
  const find = () => doc.passages.findIndex((item) => String(item.text || "").replace(/\s+/g, " ").includes(text));
  let index = find();
  if (index < 0 && doc.offset + doc.passages.length < doc.total) {
    const more = await api.document(doc.id, { offset: doc.offset + doc.passages.length, limit: 200 });
    doc.passages = doc.passages.concat(more.passages || []);
    paintPreview();
    index = find();
  }
  return index < 0 ? null : index;
}

async function renderDocument() {
  const page = $("#page-document");
  const id = state.activeDocumentId;
  if (!id) {
    page.innerHTML = `<div class="page-content"><p class="empty-note">Choose a document from the knowledge base.</p></div>`;
    return;
  }
  let doc;
  try {
    doc = await api.document(id, { focus: state.focusPassage || "" });
  } catch (err) {
    const gone = err instanceof ApiError && [404, 422].includes(err.status);
    page.innerHTML = `
      <div class="page-content not-found">
        <div class="section-eyebrow">Document</div>
        <h1>${gone ? "This document isn't here anymore" : "This document couldn't be opened"}</h1>
        <p>${gone ? "It may have been deleted or replaced. Its old links stop working when that happens." : escapeHtml(err.message)}</p>
        <button class="btn primary" id="backKnowledge" type="button">Go to the knowledge base</button>
      </div>`;
    listen($("#backKnowledge"), "click", () => showPage("knowledge"));
    $("#currentCrumb").textContent = "Not found";
    setTitle("Document not found");
    return;
  }
  state.doc = { id, name: doc.name, passages: doc.passages || [], total: doc.passage_count || 0, offset: doc.offset || 0 };
  const title = docTitle(doc.name);
  $("#currentCrumb").textContent = truncate(title, 40);
  setTitle(title);
  const downloadsOff = state.settings?.allow_downloads === false;
  const canOpen = doc.downloadable && !downloadsOff;
  const openNote = canOpen ? "" : doc.downloadable ? "Opening originals is turned off in Settings." : "The original file wasn't kept for this document, so it can't be opened.";
  page.innerHTML = `
    <div class="page-content">
      <header class="page-header doc-header">
        <div>
          <button class="back-link" id="backKnowledge" type="button">← Knowledge base</button>
          <h1>${escapeHtml(title)}</h1>
          <p>${escapeHtml(doc.name)} · ${escapeHtml(doc.collection || "General")} collection · ${(doc.chunk_count || 0).toLocaleString()} chunks${doc.max_page ? ` · ${doc.max_page} page${doc.max_page === 1 ? "" : "s"}` : ""}${doc.uploaded_at ? ` · added ${escapeHtml(whenInline(doc.uploaded_at))}` : ""}</p>
        </div>
        <div class="page-actions-stack">
          <div class="page-actions">
            <button class="btn" id="askThis" type="button">Ask about this</button>
            <button class="btn" id="replaceFile" type="button" data-owner>Replace file</button>
            <button class="btn dark" id="openOriginal" type="button" ${canOpen ? "" : "disabled"}>Open original ↗</button>
            <button class="btn danger" id="deleteDoc" type="button" data-owner>Delete</button>
          </div>
          ${openNote ? `<p class="action-note">${openNote}</p>` : ""}
        </div>
      </header>
      <div class="doc-layout">
        <article class="panel document-preview" id="documentPreview">
          <div class="preview-tools">
            <label class="filter-field"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></svg><input id="docFind" type="search" placeholder="Find in this document" aria-label="Find in this document" /></label>
            <span class="find-count" id="findCount" aria-live="polite"></span>
          </div>
          <div id="previewBody"></div>
          <div class="preview-more" id="previewMore"></div>
        </article>
        <aside class="stack">
          <section class="panel" data-owner>
            <div class="panel-head"><div><h2>Index record</h2><p>Saved with this document</p></div></div>
            <div class="form-block">
              <div class="switch-row"><div><strong id="incLabel">Included in answers</strong><p>When off, retrieval skips this file</p></div><button type="button" class="switch ${doc.included ? "on" : ""}" id="includedSwitch" role="switch" aria-checked="${doc.included ? "true" : "false"}" aria-labelledby="incLabel"></button></div>
              <div class="switch-row"><div><strong id="citeLabel">Citation required</strong><p>Answers drawing on this file must cite a source</p></div><button type="button" class="switch ${doc.citation_required ? "on" : ""}" id="citeSwitch" role="switch" aria-checked="${doc.citation_required ? "true" : "false"}" aria-labelledby="citeLabel"></button></div>
            </div>
          </section>
          <section class="panel">
            <div class="panel-head"><div><h2>Indexed passages</h2><p>${(doc.chunk_count || 0).toLocaleString()} chunks · ${doc.passage_count ?? 0} parent passages · ${escapeHtml(state.index?.chunking || "Parent-child")}</p></div></div>
            <div class="chunk-list" id="chunkList"></div>
          </section>
        </aside>
      </div>
    </div>`;
  paintPreview();
  listen($("#backKnowledge"), "click", () => showPage("knowledge"));
  listen($("#askThis"), "click", () => {
    state.scopeId = id;
    updateScopeLabel();
    startNewConversation();
  });
  listen($("#replaceFile"), "click", () => chooseFile("replace", { replaceId: doc.id }));
  listen($("#openOriginal"), "click", () => openOriginal(doc));
  listen($("#deleteDoc"), "click", () => confirmDelete(doc));
  listen($("#includedSwitch"), "click", (event) =>
    togglePolicy(event.currentTarget, doc.id, "included", "Document included in answers", "Document excluded from answers")
  );
  listen($("#citeSwitch"), "click", (event) =>
    togglePolicy(event.currentTarget, doc.id, "citation_required", "Citations are required for this document", "Citations are optional for this document")
  );
  const find = $("#docFind");
  find.addEventListener("input", () => highlightFind(find.value));
  find.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      jumpFind(event.shiftKey ? -1 : 1);
    }
  });
  let focus = doc.focus_index != null ? doc.focus_index - state.doc.offset : null;
  if (focus == null && state.focusProbe) focus = await focusByText(state.focusProbe);
  state.focusProbe = null;
  if (focus != null) {
    const target = $(`#previewBody [data-preview="${focus}"]`);
    if (target) {
      target.classList.add("focus", "flash");
      $(`#chunkList [data-passage="${focus}"]`)?.classList.add("active");
      target.scrollIntoView({ block: "center" }); // a jump, not a long smooth scroll; the flash shows where you landed
    }
  }
}

function paintPreview() {
  const doc = state.doc;
  if (!doc) return;
  $("#previewBody").innerHTML = documentPreview(doc.passages);
  const shown = doc.passages.length;
  const end = doc.offset + shown;
  $("#previewMore").innerHTML =
    end < doc.total
      ? `<p class="empty-note">Showing passages ${doc.offset + 1}–${end} of ${doc.total}.</p><button type="button" class="btn" id="loadMore">Load more passages</button>`
      : doc.offset
        ? `<p class="empty-note">Showing passages ${doc.offset + 1}–${end} of ${doc.total}.</p>`
        : "";
  $("#chunkList").innerHTML =
    doc.passages
      .map(
        (item, index) =>
          `<button type="button" class="chunk" data-passage="${index}"><span>Passage ${String(doc.offset + index + 1).padStart(3, "0")} · page ${item.page_number || "—"}${item.header_context ? ` · ${escapeHtml(item.header_context.split(" > ").pop())}` : ""}</span><p>${escapeHtml(truncate(String(item.text || "").replace(/^#{1,6}\s+/gm, "").replace(/\s+/g, " "), 180))}</p></button>`
      )
      .join("") || `<p class="empty-note">No passages stored.</p>`;
  listen($("#loadMore"), "click", async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    button.textContent = "Loading…";
    const more = await api.document(doc.id, { offset: doc.offset + doc.passages.length, limit: 40 });
    doc.passages = doc.passages.concat(more.passages || []);
    paintPreview();
    highlightFind($("#docFind").value);
  });
  $$("#chunkList [data-passage]").forEach((button) =>
    listen(button, "click", () => {
      $$("#chunkList [data-passage]").forEach((item) => item.classList.toggle("active", item === button));
      $$("#previewBody .preview-passage").forEach((item) => item.classList.remove("focus", "flash"));
      const target = $(`#previewBody [data-preview="${button.dataset.passage}"]`);
      if (!target) return;
      target.classList.add("focus", "flash");
      target.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "center" });
    })
  );
}

// Render stored passages as a readable preview: Markdown-style headings become headings,
// tables and plain lines stay text, links work. Each passage is addressable so a chunk can be focused.
function documentPreview(passages) {
  if (!passages.length) return `<p class="empty-note">No extractable preview.</p>`;
  let lastPage = null;
  return passages
    .map((passage, index) => {
      const lines = String(passage.text || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
      const body = lines
        .map((line) => {
          const heading = /^(#{1,4})\s+(.*)$/.exec(line);
          if (heading) return heading[1].length <= 1 ? `<h2>${escapeHtml(heading[2])}</h2>` : `<h3>${escapeHtml(heading[2])}</h3>`;
          if (/^\|.*\|$/.test(line)) return /^\|[\s|:-]+\|$/.test(line) ? "" : `<p class="preview-table">${escapeHtml(line)}</p>`;
          return `<p>${linkify(escapeHtml(line.replace(/^[-*]\s+/, "• ")))}</p>`;
        })
        .join("");
      const pageMark = passage.page_number !== lastPage ? `<div class="section-eyebrow page-mark">Page ${escapeHtml(passage.page_number || "—")}</div>` : "";
      lastPage = passage.page_number;
      return `${pageMark}<section class="preview-passage" data-preview="${index}">${body}</section>`;
    })
    .join("");
}

let findIndex = -1;

function highlightFind(query) {
  const root = $("#previewBody");
  if (!root) return;
  $$("mark.find", root).forEach((mark) => mark.replaceWith(document.createTextNode(mark.textContent)));
  root.normalize();
  findIndex = -1;
  const q = query.trim();
  const count = $("#findCount");
  if (!q) {
    count.textContent = "";
    return;
  }
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const hits = [];
  const lower = q.toLowerCase();
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (node.nodeValue.toLowerCase().includes(lower)) hits.push(node);
  }
  let total = 0;
  for (const node of hits) {
    const text = node.nodeValue;
    const fragment = document.createDocumentFragment();
    let at = 0;
    let i = text.toLowerCase().indexOf(lower);
    while (i !== -1) {
      fragment.appendChild(document.createTextNode(text.slice(at, i)));
      const mark = document.createElement("mark");
      mark.className = "find";
      mark.textContent = text.slice(i, i + q.length);
      fragment.appendChild(mark);
      total += 1;
      at = i + q.length;
      i = text.toLowerCase().indexOf(lower, at);
    }
    fragment.appendChild(document.createTextNode(text.slice(at)));
    node.replaceWith(fragment);
  }
  const doc = state.doc;
  const partial = doc && doc.offset + doc.passages.length < doc.total ? " in loaded passages" : "";
  count.textContent = total ? `${total} match${total === 1 ? "" : "es"}${partial} · Enter for next` : `No matches${partial}`;
  if (total) jumpFind(1);
}

function jumpFind(step) {
  const marks = $$("#previewBody mark.find");
  if (!marks.length) return;
  marks[findIndex]?.classList.remove("current");
  findIndex = (findIndex + step + marks.length) % marks.length;
  marks[findIndex].classList.add("current");
  marks[findIndex].scrollIntoView({ block: "center", behavior: reducedMotion() ? "auto" : "smooth" });
}

async function togglePolicy(button, id, key, onText, offText) {
  const next = !button.classList.contains("on");
  button.disabled = true;
  try {
    await api.updateDocument(id, { [key]: next });
    button.classList.toggle("on", next);
    button.setAttribute("aria-checked", next ? "true" : "false");
    notify(next ? onText : offText, { type: "success" });
    await refreshShell();
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

async function confirmDelete(doc) {
  const ok = await confirmDialog({
    title: `Delete ${doc.name}?`,
    body: "This removes the file, its vectors, and its keyword index. Answers will no longer cite it. This can't be undone.",
    confirmLabel: "Delete document",
    danger: true,
  });
  if (!ok) return;
  await api.deleteDocument(doc.id);
  if (state.activeDocumentId === doc.id) state.activeDocumentId = null;
  if (state.scopeId === doc.id) state.scopeId = null;
  updateScopeLabel();
  notify(`${doc.name} deleted`, { type: "success" });
  if (state.page === "knowledge") await renderKnowledge();
  else showPage("knowledge");
}

// ---------------------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------------------

function pipeRow(nodes, packet = false) {
  return `<div class="pipeline-row">${packet ? '<i class="packet" aria-hidden="true"></i>' : ""}${nodes.join(pipeArrow)}</div>`;
}

async function renderPipeline() {
  const [index, settings, evals] = await Promise.all([api.index(), ensureSettings(), api.evalLatest().catch(() => ({}))]);
  state.index = index;
  state.settings = settings;
  const first = !$("#page-pipeline .stat-card");
  const stats = index.stats || {};
  const drift = settings.settings_drift || {};
  const driftKeys = Object.keys(drift);
  const pretty = { top_k: "top-k", similarity_threshold: "similarity floor", max_parents: "context limit", rrf_k: "fusion constant" };
  const driftNote = driftKeys.length
    ? `<div class="settings-drift-warning" role="status"><strong>Your saved search settings differ from this server's defaults</strong><p>Questions use ${driftKeys
        .map((key) => `${escapeHtml(pretty[key] || key)} ${escapeHtml(drift[key].saved)} (default ${escapeHtml(drift[key].env_default)})`)
        .join(", ")}. Change them in Settings → Retrieval.</p><details><summary>For developers</summary><p>The eval pins its own values in <code>backend/eval/config.json</code>, so eval results may not match what questions in this workspace use.</p></details></div>`
    : "";
  const version = escapeHtml(String(index.version_id || "").slice(0, 8) || "—");
  const jevState = index.jev_circuit === "open" ? "Degraded" : index.jev_configured ? "Ready" : "No key";
  const latestEval = evals?.hermetic || null;
  const node = (label, title, detail, on = false) =>
    `<div class="pipe-node ${on ? "on" : ""}"><span class="node-label">${label}</span><strong>${title}</strong><small>${detail}</small></div>`;
  const health = stats.run_success_rate == null ? null : Math.round(Number(stats.run_success_rate) * 1000) / 10;
  $("#page-pipeline").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Index architecture</div><h1>Pipeline</h1><p>Ingestion, retrieval, Jev reranking, and the grounding check for the active index.</p></div>
        <div class="page-actions">
          <button class="btn" id="viewRegistry" type="button">Version history</button>
          <button class="btn" id="reconcileIndex" type="button" data-owner>Reconcile</button>
          <button class="btn primary" id="refreshIndex" type="button" data-owner>${ICON.retry}Refresh index</button>
        </div>
      </header>
      ${driftNote}
      <div class="stat-grid">
        <article class="stat-card"><span>Active version</span><div class="stat-value">${version}</div><small>${escapeHtml(index.embedding_model || "")} · ${index.embedding_dimensions ?? "—"} dimensions</small></article>
        <article class="stat-card accent"><span>Pipeline health</span><div class="stat-value" ${health == null ? "" : `data-count="${health}" data-format="pct"`}>${health == null ? "—" : `${health}%`}</div><small>${stats.runs_counted ? `${stats.runs_counted} runs in 30 days` : "No runs in 30 days"}</small></article>
        <article class="stat-card"><span>Mean retrieval</span><div class="stat-value">${stats.mean_retrieval_ms == null ? "—" : escapeHtml(seconds(stats.mean_retrieval_ms))}</div><small>Search + rerank, last 30 days</small></article>
        <article class="stat-card dark"><span>Last handoff</span><div class="stat-value">${stats.last_handoff_at ? escapeHtml(when(stats.last_handoff_at)) : "—"}</div><small>${(index.chunks ?? 0).toLocaleString()} chunks in the active index</small></article>
      </div>
      <div class="two-column">
        <section class="panel">
          <div class="panel-head"><div><h2>Live architecture</h2><p>What a question actually runs</p></div><span class="status ${index.compatible ? "" : "sync"}">${index.compatible ? "All systems normal" : "Needs refresh"}</span></div>
          <div class="pipeline-map">
            <div class="branch-label">Ingestion path</div>
            ${pipeRow([
              node("Source", "Documents", `${index.documents ?? 0} active`, true),
              node("Prepare", `${escapeHtml(index.chunking || "Parent-child")} chunking`, `${(index.chunks ?? 0).toLocaleString()} chunks`, true),
              node("Encode", "Tokenize + embed", `${index.embedding_dimensions ?? "—"} dims · ${escapeHtml(index.embed_style || "raw")}`, true),
              node("Store", "Vector + keyword index", `${version} active`, true),
            ], true)}
            <div class="branch-label" style="margin-top:38px">Query path</div>
            ${pipeRow([
              node("Input", "Condense + embed", "follow-ups rewritten"),
              node("Retrieve", "Hybrid search", `k = ${settings.top_k} · floor ${Number(settings.similarity_threshold).toFixed(2)}`),
              node("Refine", "Jev reranker", `${escapeHtml(jevState)} · keep ≥ ${Number(index.jev_relevance_threshold ?? 0.2).toFixed(2)}`),
              node("Answer", "Ground + validate", latestEval?.metrics?.answers ? `${percent(latestEval.metrics.answers.answer_rate)} eval answer rate` : escapeHtml(index.answer_model || "OpenRouter")),
            ], true)}
            <div class="pipeline-legend"><span><i></i> Production active</span><span>Jev via OpenRouter · ${escapeHtml(index.jev_model || "")}</span></div>
          </div>
        </section>
        <div class="stack">
          <section class="panel">
            <div class="panel-head"><div><h2>Recent runs</h2><p>Uploads, deletes, and handoffs</p></div></div>
            <div class="run-list">${(index.runs || []).slice(0, 6).map((run) => `<div class="run"><span class="run-time">${escapeHtml(stamp(run.created_at))}</span><div><strong>${escapeHtml(run.name)}</strong><p>${escapeHtml(run.detail)}</p></div><span class="status ${run.status === "Success" ? "" : "sync"}">${escapeHtml(run.status)}</span></div>`).join("") || `<p class="empty-note">No runs recorded yet.</p>`}</div>
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
            <div class="panel-head"><div><h2>Quality gate</h2><p>${qualityCaption(latestEval)}</p></div>${qualityStatus(latestEval)}</div>
            ${evalPanel(latestEval)}
          </section>
        </div>
      </div>
    </div>`;
  listen($("#viewRegistry"), "click", () => showRegistry());
  listen($("#reconcileIndex"), "click", async () => {
    const ok = await confirmDialog({
      title: "Reconcile the index?",
      body: "This removes index records whose documents no longer exist, so stale passages can't be retrieved. Documents and their files are not touched.",
      confirmLabel: "Reconcile",
    });
    if (!ok) return;
    const result = await api.reconcileIndex();
    const orphans = result.orphaned_documents || [];
    notify(
      `Reconcile removed ${result.removed} orphaned record${result.removed === 1 ? "" : "s"}` +
        (orphans.length ? `; ${orphans.length} indexed document${orphans.length === 1 ? " is" : "s are"} missing from the catalog` : ""),
      { type: "success" }
    );
    await renderPipeline();
  });
  listen($("#refreshIndex"), "click", () => refreshIndex());
  if (first) countUp($("#page-pipeline"));
  setTitle("Index pipeline");
}

// The quality gate is Needle's built-in exam: test questions with known answers, run before a change
// ships. A change that makes retrieval or answers worse fails the gate.
function qualityCaption(report) {
  if (!report) return "Not measured on this server yet";
  const n = report.questions || report.metrics?.questions;
  const day = report.finished_at ? dayLabel(new Date(report.finished_at)) : "";
  return report.source === "baseline"
    ? `Release baseline · ${n || 117} test questions${day ? ` · ${day}` : ""}`
    : `Last test run${n ? ` · ${n} questions` : ""}${day ? ` · ${day}` : ""}`;
}

function qualityStatus(report) {
  if (!report) return "";
  if (report.source === "baseline") return `<span class="status">Baseline</span>`;
  if (!report.gate) return "";
  return `<span class="status ${report.gate.passed ? "" : "sync"}">${report.gate.passed ? "Passed" : "Failed"}</span>`;
}

function evalPanel(report) {
  if (!report) {
    return `<div class="form-block quality-empty">
      <p>The quality gate is a built-in exam: test questions with known answers that Needle must get right before a change ships. Results appear here after a test run.</p>
      ${isOwner() ? `<details class="dev-note"><summary>How to run it</summary><p>On the server, in <code>backend/</code>: <code>python -m eval run</code></p></details>` : ""}
    </div>`;
  }
  const retrieval = report.metrics?.retrieval || {};
  const answers = report.metrics?.answers;
  const abstain = report.metrics?.abstention;
  const rows = [
    ["Right passage found", "The correct passage was in the top five results", percent(retrieval.hit_at_5)],
    ["Ranked first", "How often the correct passage came first", percent(retrieval.hit_at_1)],
  ];
  if (answers) {
    rows.push(["Answered", "Answerable questions released with sources", percent(answers.answer_rate)]);
    rows.push(["Facts correct", "Required facts present in released answers", percent(answers.key_fact_recall)]);
    rows.push(["Declined correctly", "Unanswerable questions held back instead of guessed", percent(abstain?.recall)]);
  }
  return `<div class="metric-list">${rows
    .map(([label, hint, value]) => `<div class="metric-row"><div><strong>${label}</strong><p>${hint}</p></div><div class="metric-score">${value}</div></div>`)
    .join("")}</div>`;
}

// A refresh builds a new version beside the live one; the server answers when it's done.
async function refreshIndex(override = false) {
  const started = Date.now();
  openModal(
    "Refreshing the index",
    `<p>Needle builds a new index version beside the live one and switches over only if it's complete and search quality holds up. Questions keep working while this runs.</p>
     <ol class="refresh-steps"><li>Copy every passage into a new version</li><li>Embed passages with the current settings</li><li>Compare recall on your golden questions</li><li>Publish the new version</li></ol>
     <div class="refresh-progress">${loader("small")}<div class="indeterminate" role="progressbar" aria-label="Refreshing the index"><i></i></div></div>
     <p class="refresh-elapsed" id="refreshElapsed">Started just now</p>`,
    [],
    { locked: true }
  );
  const tick = setInterval(() => {
    const node = $("#refreshElapsed");
    if (node) node.textContent = `${Math.floor((Date.now() - started) / 1000)}s elapsed`;
  }, 1000);
  try {
    const status = await api.refreshIndex(override);
    clearInterval(tick);
    closeModal(true);
    notify(`Published index ${String(status.version_id || "").slice(0, 8)}`, { type: "success" });
    await refreshShell();
    await renderPipeline();
  } catch (err) {
    clearInterval(tick);
    closeModal(true);
    if (err instanceof ApiError && err.status === 409 && err.detail?.overridable) {
      const ok = await confirmDialog({
        title: "Publish without the recall check?",
        body: `${escapeHtml(err.message)} The new version is still checked for a complete copy; only the golden-question recall comparison is skipped.`,
        confirmLabel: "Publish anyway",
      });
      if (ok) await refreshIndex(true);
      return;
    }
    throw err;
  }
}

async function showRegistry() {
  const { versions = [] } = await api.indexVersions();
  const rows = versions
    .map(
      (item) => `<tr><td><strong>${escapeHtml(String(item.version_id).slice(0, 8))}</strong><br><small>${escapeHtml(item.collection_name)}</small></td><td><span class="status ${item.status === "active" ? "" : "sync"}">${escapeHtml(item.status)}</span></td><td>${escapeHtml(item.embedding_model)}<br><small>${escapeHtml(item.embed_style)} · ${escapeHtml(item.chunking)}</small></td><td>${item.chunk_count ?? "—"}</td><td>${escapeHtml(stamp(item.activated_at || item.created_at))}</td></tr>`
    )
    .join("");
  const canRollBack = isOwner() && versions.some((item) => item.status === "retired");
  openModal(
    "Version history",
    `<div class="table-scroll"><table class="data-table"><thead><tr><th>Version</th><th>Status</th><th>Model</th><th>Chunks</th><th>When</th></tr></thead><tbody>${rows || `<tr><td colspan="5"><p class="empty-note">No versions recorded.</p></td></tr>`}</tbody></table></div>`,
    [
      { label: "Close", onClick: () => closeModal() },
      ...(canRollBack
        ? [
            {
              label: "Roll back to previous",
              danger: true,
              onClick: async () => {
                closeModal();
                const ok = await confirmDialog({
                  title: "Roll back the index?",
                  body: "The previous version becomes live again. Documents added since it was built won't be searchable until you refresh.",
                  confirmLabel: "Roll back",
                  danger: true,
                });
                if (!ok) return;
                const restored = await api.rollbackIndex();
                notify(`Restored index ${String(restored.version_id || "").slice(0, 8)}`, { type: "success" });
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

// ---------------------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------------------

async function renderAnalytics() {
  const [report, evals] = await Promise.all([api.analytics(state.analyticsDays), api.evalLatest().catch(() => ({}))]);
  const first = !$("#page-analytics .stat-card");
  const series = report.series || [];
  const prior = report.prior || {};
  const max = Math.max(1, ...series.map((point) => Number(point.questions) || 0));
  const gaps = report.gaps || [];
  const step = Math.max(1, Math.ceil(series.length / 6));
  // Date labels sit under their own bars (and always include the most recent day).
  const labelAt = (i) => `<span style="left:${(((i + 0.5) / series.length) * 100).toFixed(2)}%">${escapeHtml(shortDay(series[i].day))}</span>`;
  const labelIdx = series.map((_, i) => i).filter((i) => (series.length - 1 - i) % step === 0);
  const labels = labelIdx.map(labelAt).join("");
  const ticks = [...new Set(max <= 4 ? Array.from({ length: max + 1 }, (_, i) => i) : [0, Math.round(max / 2), max])];
  const perDay = report.questions ? report.questions / (report.days || state.analyticsDays) : 0;
  const dailyAverage = perDay && perDay < 0.1 ? "<0.1" : perDay.toFixed(1);
  const evalReport = evals?.hermetic;
  const abstainPrecision = evalReport?.metrics?.abstention?.precision;
  const ratings = report.ratings ?? 0;
  const fewRatings = ratings > 0 && ratings < 10;
  const pct = (value) => (value == null ? null : Math.round(Number(value) * 1000) / 10);
  const countAttr = (value) => (value == null ? "" : `data-count="${value}" data-format="pct"`);
  $("#page-analytics").innerHTML = `
    <div class="page-content">
      <header class="page-header">
        <div><div class="section-eyebrow">Quality intelligence</div><h1>Analytics</h1><p>Adoption, answer quality, and coverage gaps from questions this workspace has actually asked.</p></div>
        <div class="page-actions">
          <div class="segmented" id="range" role="group" aria-label="Time range">${[7, 30, 90].map((days) => `<button type="button" data-days="${days}" class="${days === state.analyticsDays ? "active" : ""}" aria-pressed="${days === state.analyticsDays}">${days} days</button>`).join("")}</div>
          <button class="btn" id="exportReport" type="button" data-owner>Export CSV</button>
        </div>
      </header>
      <div class="stat-grid">
        <article class="stat-card dark"><span>Questions asked</span><div class="stat-value" data-count="${report.questions ?? 0}">${(report.questions ?? 0).toLocaleString()}</div><small>${dailyAverage} a day on average</small></article>
        <article class="stat-card accent"><span>Grounded answer rate</span><div class="stat-value" ${countAttr(pct(report.grounded_rate))}>${percent(report.grounded_rate)}</div><small class="stat-delta">${delta(report.grounded_rate, prior.grounded_rate) || `${report.questions ?? 0} recorded`}</small></article>
        <article class="stat-card ${fewRatings ? "low-n" : ""}"><span>Helpful rating</span><div class="stat-value" ${ratings ? countAttr(pct(report.helpful_rate)) : ""}>${ratings ? percent(report.helpful_rate) : "—"}</div><small>${ratings ? `${ratings} answer${ratings === 1 ? "" : "s"} rated${fewRatings ? " · too few to judge yet" : ""}` : "No answers rated yet"}</small></article>
        <article class="stat-card"><span>Withheld rate</span><div class="stat-value" ${countAttr(pct(report.withheld_rate))}>${percent(report.withheld_rate)}</div><small class="stat-delta">${delta(report.withheld_rate, prior.withheld_rate, { invert: true }) || "No prior period to compare"}</small></article>
      </div>
      <div class="two-column">
        <section class="panel">
          <div class="panel-head"><div><h2>Answer volume &amp; quality</h2><p>Questions per day, and how many were released as grounded</p></div>
            <div class="chart-legend"><span><i class="key asked"></i>Asked</span><span><i class="key grounded"></i>Grounded</span></div></div>
          ${
            report.questions
              ? `<div class="chart-wrap">
                  <div class="plot" id="chart" role="img" aria-label="${state.analyticsDays}-day bar chart: ${report.questions} questions, ${percent(report.grounded_rate)} grounded">
                    ${ticks.map((t) => `<i class="grid" style="bottom:${((t / max) * 100).toFixed(2)}%"><span>${t}</span></i>`).join("")}
                    <div class="bars">${series
                      .map(
                        (point, i) =>
                          `<div class="bar-group" data-day="${escapeHtml(shortDay(point.day))}" data-asked="${point.questions}" data-grounded="${point.grounded}" style="--i:${i}"><i class="bar" style="height:${((Number(point.questions) || 0) / max) * 100}%"></i><i class="bar secondary" style="height:${((Number(point.grounded) || 0) / max) * 100}%"></i></div>`
                      )
                      .join("")}</div>
                    <div class="chart-tip" id="chartTip" hidden></div>
                  </div>
                  <div class="plot-labels">${labels}</div>
                </div>`
              : `<p class="empty-note">Ask a few questions to fill this chart.</p>`
          }
        </section>
        <section class="panel">
          <div class="panel-head"><div><h2>Quality signals</h2><p>How the system is performing</p></div></div>
          <div class="metric-list">
            <div class="metric-row"><div><strong>Citation coverage</strong><p>Share of released answers whose text cites a numbered source</p></div><div class="metric-score">${percent(report.citation_coverage)}</div></div>
            <div class="metric-row"><div><strong>Average relevance</strong><p>Best Jev score per question</p></div><div class="metric-score">${report.mean_relevance == null ? "—" : Number(report.mean_relevance).toFixed(2)}</div></div>
            <div class="metric-row"><div><strong>Retrieval latency</strong><p>Median search + rerank time</p></div><div class="metric-score">${report.retrieval_p50_ms == null ? "—" : escapeHtml(seconds(report.retrieval_p50_ms))}</div></div>
            <div class="metric-row"><div><strong>Fallback precision</strong><p>${evalReport ? "Of the questions held back in testing, how many truly had no answer" : "Measured by the test questions"}</p></div><div class="metric-score">${percent(abstainPrecision)}</div></div>
          </div>
        </section>
      </div>
      <div class="viz-grid">
        <section class="panel viz-panel">
          <div class="panel-head"><div><h2>How questions ended</h2><p>Every question in the last ${state.analyticsDays} days, by outcome</p></div></div>
          <div class="viz-body">${outcomeBar(report.outcomes || {})}</div>
        </section>
        <section class="panel viz-panel">
          <div class="panel-head"><div><h2>Response time</h2><p>Time to an answer, per day: the median and the slowest 5%</p></div></div>
          <div class="viz-body">${lineChart({
            points: (report.latency_series || []).map((p) => ({ ...p, label: shortDay(p.day) })),
            series: [
              { key: "p50_ms", label: "Median", color: "#2a78d6" },
              { key: "p95_ms", label: "Slowest 5%", color: "#eb6834" },
            ],
            ariaLabel: "Daily median and 95th percentile response time",
            format: (ms) => (ms == null ? "—" : seconds(ms)),
          })}</div>
        </section>
        <section class="panel viz-panel">
          <div class="panel-head"><div><h2>When people ask</h2><p>Questions by hour of the day, in your time zone</p></div></div>
          <div class="viz-body">${hoursChart(report.hours_utc || [])}</div>
        </section>
        <section class="panel viz-panel">
          <div class="panel-head"><div><h2>Evidence strength</h2><p>The best relevance score each question got; low scores usually mean a coverage gap</p></div></div>
          <div class="viz-body">${columnChart({
            values: (report.relevance_bins || []).map((b) => b.count),
            labels: (report.relevance_bins || []).map((b) => `${b.from.toFixed(1)}–${b.to.toFixed(1)}`),
            tips: (report.relevance_bins || []).map((b) => [`Relevance ${b.from.toFixed(1)}–${b.to.toFixed(1)}`, `${b.count} question${b.count === 1 ? "" : "s"}`]),
            ariaLabel: "Questions by best evidence relevance",
          })}</div>
        </section>
        <section class="panel viz-panel wide">
          <div class="panel-head"><div><h2>Most-used documents</h2><p>How many answers drew on each document</p></div></div>
          <div class="viz-body">${rankBars((report.top_documents || []).map((d) => ({ label: d.name, value: d.answers })), { unit: "answer" })}</div>
        </section>
      </div>
      <section class="panel" style="margin-top:14px">
        <div class="panel-head"><div><h2>Knowledge gaps</h2><p>Questions Needle held back instead of guessing. “Not in the documents”: nothing strong enough was found, so add a source that covers it. “Failed the check”: a draft was written but not backed by the passages, so asking again (or rewording) may work.</p></div></div>
        ${state.role === "visitor" ? `<p class="empty-note gap-private">In the public demo, visitors’ questions stay private, so this list is visible only to the owner. The totals above include everyone’s questions.</p>` : `<div class="table-scroll"><table class="data-table"><thead><tr><th>Question</th><th>Why</th><th>Attempts</th><th>Best match</th><th>Last asked</th><th><span class="sr-only">Action</span></th></tr></thead><tbody>
          ${gaps.map((gap, index) => `<tr><td><strong>${escapeHtml(gap.query)}</strong></td><td>${gap.outcome === "check_failed" ? "Failed the check" : "Not in the documents"}</td><td>${gap.attempts}</td><td>${gap.best_similarity == null ? "—" : `${Number(gap.best_similarity).toFixed(2)} similarity`}</td><td>${escapeHtml(when(gap.last_seen))}</td><td><button class="btn small" type="button" data-gap="${index}" ${gap.outcome === "check_failed" ? "" : "data-owner"}>${gap.outcome === "check_failed" ? "Ask again" : "Add a source"}</button></td></tr>`).join("") || `<tr><td colspan="6"><p class="empty-note">No withheld questions in this range. Gaps appear here when a question isn’t covered by your documents or an answer fails the grounding check.</p></td></tr>`}
        </tbody></table></div>`}
      </section>
    </div>`;
  $$("#range button").forEach((button) =>
    listen(button, "click", () => {
      state.analyticsDays = Number(button.dataset.days);
      return renderAnalytics();
    })
  );
  listen($("#exportReport"), "click", () => exportReport());
  $$("#page-analytics [data-gap]").forEach((button) =>
    listen(button, "click", () => {
      const gap = gaps[Number(button.dataset.gap)];
      if (gap.outcome === "check_failed") return sendQuestion(gap.query);
      notify(`Choose a file that answers: “${truncate(gap.query, 80)}”. It will be asked again once indexed.`);
      chooseFile("upload", { forQuestion: gap.query });
    })
  );
  wireChart();
  wireTips($("#page-analytics"));
  initSegmented($("#range"));
  if (first) countUp($("#page-analytics"));
  setTitle("Analytics");
}

// Questions by local hour: the server counts UTC hours, the browser shifts them.
function hoursChart(utc) {
  const offset = Math.round(-new Date().getTimezoneOffset() / 60);
  const local = Array.from({ length: 24 }, (_, hour) => utc[(hour - offset + 48) % 24] || 0);
  const label = (h) => `${h % 12 || 12}${h < 12 ? "am" : "pm"}`;
  return columnChart({
    values: local,
    labels: local.map((_, h) => label(h)),
    tips: local.map((n, h) => [`${label(h)}–${label((h + 1) % 24)}`, `${n} question${n === 1 ? "" : "s"}`]),
    ariaLabel: "Questions by hour of the day",
    every: 6,
  });
}

function wireChart() {
  const chart = $("#chart");
  const tip = $("#chartTip");
  if (!chart || !tip) return;
  const show = (group) => {
    $$(".bar-group.on", chart).forEach((other) => other.classList.remove("on"));
    group.classList.add("on");
    tip.innerHTML = `<strong>${escapeHtml(group.dataset.day)}</strong><span><i class="key asked"></i>${group.dataset.asked} asked</span><span><i class="key grounded"></i>${group.dataset.grounded} grounded</span>`;
    tip.hidden = false;
    const half = tip.offsetWidth / 2;
    const center = group.offsetLeft + group.offsetWidth / 2;
    tip.style.left = `${Math.min(Math.max(center, half + 4), chart.clientWidth - half - 4)}px`;
  };
  const hide = () => {
    tip.hidden = true;
    $$(".bar-group.on", chart).forEach((group) => group.classList.remove("on"));
  };
  $$(".bar-group", chart).forEach((group) => {
    group.addEventListener("mouseenter", () => show(group));
    group.addEventListener("mouseleave", hide);
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

// ---------------------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------------------

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
const switchKeys = ["show_traces", "allow_downloads", "require_citations", "withhold_ungrounded", "contextual_embeddings"];

function applySettingsForm() {
  const page = $("#page-settings");
  const form = state.form;
  if (!page || !form) return;
  settingFields.forEach(([id, key]) => {
    const node = $(`#${id}`, page);
    if (node && document.activeElement !== node) node.value = form[key] ?? "";
    node?.refreshCustom?.();
  });
  $$("[data-switch]", page).forEach((button) => {
    const on = Boolean(form[button.dataset.switch]);
    button.classList.toggle("on", on);
    button.setAttribute("aria-checked", on ? "true" : "false");
  });
}

function changedSettings() {
  const saved = state.settings || {};
  const form = state.form || {};
  return [...settingFields.map(([, key]) => key), ...switchKeys].filter((key) => String(form[key] ?? "") !== String(saved[key] ?? ""));
}

function syncSaveBar() {
  const bar = $("#saveBar");
  if (!bar) return;
  const changed = changedSettings();
  state.draftDirty = changed.length > 0;
  bar.hidden = !state.draftDirty;
  $("#saveBarText").textContent = `${changed.length} unsaved change${changed.length === 1 ? "" : "s"}`;
}

function syncSettingsControl(node, key) {
  if (!state.form) state.form = { ...(state.settings || {}) };
  state.form[key] = node.type === "number" ? Number(node.value) : node.value;
  syncSaveBar();
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
    workspace_name: String(form.workspace_name ?? "").trim() || "Needle",
    profile_name: String(form.profile_name ?? "").trim() || "Operator",
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
  const switches = (name, label) =>
    `<button type="button" class="switch" data-switch="${name}" role="switch" aria-checked="false" aria-label="${escapeHtml(label)}"></button>`;
  const hint = (key, range) => `<small class="field-hint">Default ${RETRIEVAL_DEFAULTS[key]} · ${range}</small>`;
  $("#page-settings").innerHTML = `
    <div class="page-content">
      <header class="page-header"><div><div class="section-eyebrow">Workspace control</div><h1>Settings</h1><p>Saved on this server and used by the next question.</p></div></header>
      <div class="settings-layout">
        <nav class="settings-nav" aria-label="Settings sections">
          <button type="button" class="active" data-settings="general" aria-current="true">General</button>
          <button type="button" data-settings="answers">Answer behavior</button>
          <button type="button" data-settings="retrieval">Retrieval</button>
          <button type="button" data-settings="access">Access</button>
          <button type="button" data-settings="data">Data</button>
        </nav>
        <div>
          <section class="panel settings-section active" data-section="general">
            <div class="panel-head"><div><h2>General</h2><p>Workspace identity and shared preferences</p></div></div>
            <div class="form-block"><h3>Workspace profile</h3><p>Shown in the top bar and on exported reports.</p><div class="field-grid">
              <div class="field"><label for="setName">Workspace name</label><input id="setName" maxlength="80" /></div>
              <div class="field"><label for="setProfile">Your name</label><input id="setProfile" maxlength="80" /></div>
              <div class="field"><label for="setCollection">Default collection</label><select id="setCollection">${optionList(["General", "Product", "Security", "Research"], state.form.default_collection)}</select><small class="field-hint">New uploads are filed under this collection.</small></div>
            </div></div>
            <div class="form-block"><h3>Workspace preferences</h3><p>Shared behavior for everyone in this workspace.</p>
            <div class="switch-row"><div><strong>Show retrieval traces</strong><p>Show the retrieval and reranking trace next to each answer</p></div>${switches("show_traces", "Show retrieval traces")}</div>
            <div class="switch-row"><div><strong>Allow opening originals</strong><p>Open or download original files from a document's page</p></div>${switches("allow_downloads", "Allow opening originals")}</div>
            </div>
          </section>
          <section class="panel settings-section" data-section="answers">
            <div class="panel-head"><div><h2>Answer behavior</h2><p>How answers are written and when they're held back</p></div></div>
            <div class="form-block">
              <div class="field-grid">
                <div class="field"><label for="setLength">Default answer length</label><select id="setLength">${optionList(["Concise", "Balanced", "Detailed"], state.form.answer_length)}</select></div>
                <div class="field"><label for="setCiteStyle">Citation style</label><select id="setCiteStyle">${optionList(["Inline numbered", "Footnotes", "Source cards"], state.form.citation_style)}</select></div>
              </div>
              <div class="switch-row"><div><strong>Require citations</strong><p>Ask the model to cite passages as [1], [2]</p></div>${switches("require_citations", "Require citations")}</div>
              <div class="switch-row"><div><strong>No-answer fallback</strong><p>Withhold drafts that fail the grounding check</p></div>${switches("withhold_ungrounded", "No-answer fallback")}</div>
            </div>
          </section>
          <section class="panel settings-section" data-section="retrieval">
            <div class="panel-head"><div><h2>Retrieval</h2><p>Search settings apply to the next question</p></div><button type="button" class="btn small" id="resetRetrieval">Reset to defaults</button></div>
            <div class="form-block"><div class="field-grid">
              <div class="field"><label for="setTopK">Top-k candidates</label><input id="setTopK" type="number" min="1" max="100" />${hint("top_k", "1–100")}</div>
              <div class="field"><label for="setSim">Similarity threshold</label><input id="setSim" type="number" min="0" max="1" step="0.01" />${hint("similarity_threshold", "0–1")}</div>
              <div class="field"><label for="setParents">Reranked context limit</label><input id="setParents" type="number" min="1" max="12" />${hint("max_parents", "1–12")}</div>
              <div class="field"><label for="setRrf">Fusion constant</label><input id="setRrf" type="number" min="1" max="200" />${hint("rrf_k", "1–200")}</div>
              <div class="field"><label for="setChunking">Chunking strategy</label><select id="setChunking">${optionList(["Parent-child", "Fixed window", "Index card summary"], state.form.chunking)}</select><small class="field-hint">Takes effect when you refresh the index.</small></div>
            </div>
            <div class="switch-row"><div><strong>Heading-aware embeddings</strong><p>The next index refresh embeds the section title with each passage</p></div>${switches("contextual_embeddings", "Heading-aware embeddings")}</div>
            <div class="subtle-note">Chunking and heading-aware embeddings take effect when you refresh the index (Pipeline → Refresh index).</div>
            </div>
          </section>
          <section class="panel settings-section" data-section="access">
            <div class="panel-head"><div><h2>Access</h2><p>${session.auth_disabled ? "Sign-in is turned off on this server" : "Signed in with the workspace access token"}</p></div><button class="btn small" id="signOut" type="button" ${session.auth_disabled ? "disabled" : ""}>Sign out</button></div>
            <div class="form-block"><h3>How access works</h3><p>Everyone who opens this workspace signs in with the same access token, set as <code>NEEDLE_ACCESS_TOKEN</code> on the server (or generated on first start and stored in the data folder). Sessions last 12 hours. To revoke every session, change the token and <code>NEEDLE_SESSION_SECRET</code>, then restart.</p></div>
          </section>
          <section class="panel settings-section" data-section="data">
            <div class="panel-head"><div><h2>Local data</h2><p>This workspace runs on this machine</p></div></div>
            <div class="form-block danger-zone"><h3>Delete workspace data</h3><p>Removes indexed documents, stored files, conversations, and saved settings. This can't be undone.</p><button class="btn danger" id="resetWorkspace" type="button">Delete workspace data…</button></div>
          </section>
          <div class="save-bar" id="saveBar" hidden role="region" aria-label="Unsaved changes">
            <span id="saveBarText">Unsaved changes</span>
            <div><button class="btn" id="discardSettings" type="button">Discard</button><button class="btn primary" id="saveSettings" type="button">Save changes</button></div>
          </div>
        </div>
      </div>
    </div>`;
  const page = $("#page-settings");
  $$("select", page).forEach(enhanceSelect);
  applySettingsForm();
  syncSaveBar();
  $$(".settings-nav button", page).forEach((button) =>
    listen(button, "click", () => {
      $$(".settings-nav button", page).forEach((item) => {
        item.classList.toggle("active", item === button);
        if (item === button) item.setAttribute("aria-current", "true");
        else item.removeAttribute("aria-current");
      });
      $$(".settings-section", page).forEach((section) => section.classList.toggle("active", section.dataset.section === button.dataset.settings));
      applySettingsForm();
    })
  );
  settingFields.forEach(([id, key]) => {
    const node = $(`#${id}`, page);
    node.addEventListener("input", () => syncSettingsControl(node, key));
    node.addEventListener("change", () => syncSettingsControl(node, key));
  });
  $$("[data-switch]", page).forEach((button) =>
    button.addEventListener("click", () => {
      const on = !button.classList.contains("on");
      button.classList.toggle("on", on);
      button.setAttribute("aria-checked", on ? "true" : "false");
      if (!state.form) state.form = { ...(state.settings || {}) };
      state.form[button.dataset.switch] = on;
      syncSaveBar();
    })
  );
  listen($("#resetRetrieval"), "click", () => {
    Object.assign(state.form, RETRIEVAL_DEFAULTS);
    applySettingsForm();
    syncSaveBar();
  });
  listen($("#discardSettings"), "click", () => {
    state.form = { ...state.settings };
    applySettingsForm();
    syncSaveBar();
  });
  listen($("#saveSettings"), "click", () => saveSettings());
  listen($("#resetWorkspace"), "click", () => resetWorkspace());
  listen($("#signOut"), "click", () => signOut());
  setTitle("Settings");
}

async function saveSettings() {
  const payload = settingsPayload();
  const button = $("#saveSettings");
  button.disabled = true;
  button.textContent = "Saving…";
  try {
    const saved = await api.saveSettings(payload);
    state.settings = saved;
    state.form = { ...saved };
    applySettingsForm();
    button.textContent = "Saved ✓";
    button.classList.add("saved");
    await refreshShell();
    setTimeout(syncSaveBar, 900);
  } finally {
    setTimeout(() => {
      button.disabled = false;
      button.textContent = "Save changes";
      button.classList.remove("saved");
    }, 900);
  }
}

window.addEventListener("beforeunload", (event) => {
  if (state.draftDirty && state.page === "settings") {
    event.preventDefault();
    event.returnValue = "";
  }
});

async function resetWorkspace() {
  const name = state.settings?.workspace_name || "Needle";
  const ok = await confirmDialog({
    title: "Delete all workspace data?",
    body: "Indexed documents, stored files, conversations, and settings will be removed. This can't be undone.",
    confirmLabel: "Delete everything",
    danger: true,
    typed: name,
  });
  if (!ok) return;
  await api.reset();
  Object.assign(state, { conversationId: null, messages: [], activeTurn: null, scopeId: null, form: null, draftDirty: false, activeDocumentId: null });
  $("#threadSearch").value = "";
  notify("Workspace data removed", { type: "success" });
  await refreshShell();
  renderConversation();
  showPage("ask", { force: true });
}

// ---------------------------------------------------------------------------------------
// Command menu (Ctrl/⌘ K)
// ---------------------------------------------------------------------------------------

let paletteItems = [];
let paletteIndex = 0;

function buildPaletteItems() {
  const items = [
    { group: "Pages", label: "Ask", run: () => showPage("ask") },
    { group: "Pages", label: "Knowledge base", run: () => showPage("knowledge") },
    { group: "Pages", label: "Index pipeline", run: () => showPage("pipeline") },
    { group: "Pages", label: "Analytics", run: () => showPage("analytics") },
  ];
  if (isOwner()) items.push({ group: "Pages", label: "Settings", run: () => showPage("settings") });
  items.push({ group: "Actions", label: "New conversation", run: () => startNewConversation() });
  if (isOwner()) items.push({ group: "Actions", label: "Upload files", run: () => chooseFile("upload") });
  state.documents
    .filter((doc) => doc.status === "indexed")
    .forEach((doc) => items.push({ group: "Documents", label: docTitle(doc.name), hint: doc.name, run: () => openDocument(doc.id) }));
  state.conversations
    .filter((thread) => !state.deletedThreads.has(thread.id))
    .forEach((thread) => items.push({ group: "Conversations", label: thread.title, hint: when(thread.updated_at), run: () => openConversation(thread.id) }));
  return items;
}

function openPalette() {
  closeMenu();
  closeContextMenu();
  const root = $("#palette");
  root.hidden = false;
  requestAnimationFrame(() => root.classList.add("open"));
  const input = $("#paletteInput");
  input.value = "";
  paletteItems = buildPaletteItems();
  paintPalette();
  input.focus();
}

function closePalette() {
  const root = $("#palette");
  if (root.hidden) return false;
  root.classList.remove("open");
  root.hidden = true;
  return true;
}

function paletteMatches() {
  const words = $("#paletteInput").value.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) {
    const counts = {};
    return paletteItems.filter((item) => {
      counts[item.group] = (counts[item.group] || 0) + 1;
      return item.group === "Pages" || item.group === "Actions" || counts[item.group] <= 4;
    });
  }
  return paletteItems.filter((item) => words.every((word) => `${item.label} ${item.hint || ""} ${item.group}`.toLowerCase().includes(word))).slice(0, 30);
}

function paintPalette() {
  const list = $("#paletteList");
  const matches = paletteMatches();
  paletteIndex = Math.min(paletteIndex, Math.max(0, matches.length - 1));
  if (!matches.length) {
    list.innerHTML = `<p class="empty-note">Nothing matches. Try a document or conversation name.</p>`;
    return;
  }
  let lastGroup = "";
  list.innerHTML = matches
    .map((item, i) => {
      const head = item.group !== lastGroup ? `<div class="palette-group">${item.group}</div>` : "";
      lastGroup = item.group;
      return `${head}<button type="button" role="option" class="palette-item ${i === paletteIndex ? "active" : ""}" aria-selected="${i === paletteIndex}" data-index="${i}"><span>${escapeHtml(truncate(item.label, 70))}</span>${item.hint ? `<small>${escapeHtml(truncate(item.hint, 40))}</small>` : ""}</button>`;
    })
    .join("");
  $$(".palette-item", list).forEach((button) =>
    button.addEventListener("click", () => runPalette(matches[Number(button.dataset.index)]))
  );
  $(".palette-item.active", list)?.scrollIntoView({ block: "nearest" });
}

function runPalette(item) {
  if (!item) return;
  closePalette();
  Promise.resolve(item.run()).catch((err) => notify(err.message, { type: "error" }));
}

$("#paletteInput").addEventListener("input", () => {
  paletteIndex = 0;
  paintPalette();
});
$("#paletteInput").addEventListener("keydown", (event) => {
  const matches = paletteMatches();
  if (event.key === "ArrowDown") {
    event.preventDefault();
    paletteIndex = (paletteIndex + 1) % Math.max(1, matches.length);
    paintPalette();
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    paletteIndex = (paletteIndex - 1 + matches.length) % Math.max(1, matches.length);
    paintPalette();
  } else if (event.key === "Enter") {
    event.preventDefault();
    runPalette(matches[paletteIndex]);
  } else if (event.key === "Tab") {
    event.preventDefault();
  }
});
$("#palette").addEventListener("click", (event) => {
  if (event.target.id === "palette") closePalette();
});

// ---------------------------------------------------------------------------------------
// Account, sign-in, sign-out
// ---------------------------------------------------------------------------------------

function accountMenu() {
  const anchor = $("#avatar");
  const settings = state.settings || {};
  const items = [];
  if (isOwner()) items.push({ label: "Settings", onClick: () => showPage("settings") });
  if (state.role === "visitor") items.push({ label: "Owner sign-in", onClick: () => showSignIn("", { optional: true }) });
  else items.push({ label: "Sign out", danger: true, onClick: () => signOut() });
  openMenu(anchor, items);
  const menu = openMenuState?.menu;
  if (menu) {
    const head = document.createElement("div");
    head.className = "menu-head";
    head.innerHTML = `<strong>${escapeHtml(settings.profile_name || "Operator")}</strong><span>${escapeHtml(state.role === "visitor" ? "Visitor · public demo" : settings.workspace_name || "Needle")}</span>`;
    menu.prepend(head);
  }
}

let signInOpen = false;

function showSignIn(message = "", { optional = false } = {}) {
  if (signInOpen && !optional) return;
  signInOpen = !optional;
  openModal(
    optional ? "Owner sign-in" : "Sign in to Needle",
    `<p>${optional ? "Enter the owner access token to upload documents and change settings." : "Enter the workspace access token. It is set as <code>NEEDLE_ACCESS_TOKEN</code> on the server, or printed in the server console on first start."}</p>${message ? `<p class="form-error" role="alert">${escapeHtml(message)}</p>` : ""}<div class="field"><label for="accessToken">Access token</label><input id="accessToken" type="password" autocomplete="current-password" /></div>`,
    [
      ...(optional ? [{ label: "Cancel", onClick: () => closeModal() }] : []),
      {
        label: "Sign in",
        primary: true,
        onClick: async () => {
          const token = $("#accessToken").value.trim();
          if (!token) {
            $("#accessToken").focus();
            return;
          }
          try {
            await api.login(token);
          } catch (err) {
            signInOpen = false;
            closeModal(true);
            showSignIn(err.message, { optional });
            return;
          }
          signInOpen = false;
          closeModal(true);
          await boot();
        },
      },
    ],
    { locked: !optional }
  );
  $("#accessToken").addEventListener("keydown", (event) => {
    if (event.key === "Enter") $("#modalActions .btn.primary")?.click();
  });
}

async function signOut() {
  const ok = await confirmDialog({ title: "Sign out?", body: "You'll need the access token to sign in again.", confirmLabel: "Sign out", danger: true });
  if (!ok) return;
  await api.logout();
  state.conversationId = null;
  state.messages = [];
  if (state.demo) {
    location.hash = "#ask";
    location.reload();
    return;
  }
  showSignIn();
}

// ---------------------------------------------------------------------------------------
// Global wiring
// ---------------------------------------------------------------------------------------

listen($("#askForm"), "submit", (event) => {
  event.preventDefault();
  if (state.busy) {
    state.pending?.controller.abort();
    return;
  }
  const value = $("#askInput").value.trim();
  if (!value) {
    $("#askInput").focus();
    return;
  }
  $("#askInput").value = "";
  $("#askInput").style.height = "auto";
  return sendQuestion(value);
});

$("#askInput").addEventListener("input", () => {
  const input = $("#askInput");
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

$("#askInput").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!state.busy) $("#askForm").requestSubmit();
  }
});

listen($("#newChat"), "click", () => startNewConversation());
listen($("#openKnowledge"), "click", () => showPage("knowledge"));
$$(".rail [data-page]").forEach((button) => listen(button, "click", () => showPage(button.dataset.page)));
listen($("#sourceToggle"), "click", () => setInspector(!$("#inspector").classList.contains("open")));
listen($("#closeInspector"), "click", () => setInspector(false));
listen($("#threadsToggle"), "click", () => setSidebar(!$("#sidebar").classList.contains("open")));
listen($("#scrim"), "click", () => {
  setInspector(false);
  setSidebar(false);
});
listen($("#paletteTrigger"), "click", () => openPalette());
listen($("#avatar"), "click", (event) => {
  event.stopPropagation();
  if (openMenuState?.anchor === $("#avatar")) closeMenu();
  else accountMenu();
});

let threadSearchSeq = 0;
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

listen($("#addContext"), "click", (event) => {
  event.stopPropagation();
  const menu = $("#contextMenu");
  if (!menu.hidden) {
    closeContextMenu();
    return;
  }
  renderContextMenu();
  menu.hidden = false;
  $("#addContext").setAttribute("aria-expanded", "true");
});

listen($("#voiceButton"), "click", () => toggleVoice());

$("#fileInput").addEventListener("change", (event) => {
  const input = event.target;
  const files = [...(input.files || [])];
  const replaceId = input.dataset.replace || null;
  const forQuestion = input.dataset.question || null;
  input.value = "";
  addFiles(files, { replaceId, forQuestion });
});

listen($("#modal"), "click", (event) => {
  if (event.target.id === "modal") closeModal();
});

document.addEventListener("click", (event) => {
  const sidebar = $("#sidebar");
  if (sidebar.classList.contains("open") && !sidebar.contains(event.target) && !$("#threadsToggle").contains(event.target)) setSidebar(false);
  const menu = $("#contextMenu");
  if (!menu.hidden && !$("#addContext").contains(event.target) && !menu.contains(event.target)) closeContextMenu();
  if (openMenuState && !openMenuState.menu.contains(event.target) && !openMenuState.anchor.contains(event.target)) closeMenu();
});

window.addEventListener("resize", () => {
  closeMenu();
  syncScrim();
});

document.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    if (!$("#modal").hidden) return;
    if ($("#palette").hidden) openPalette();
    else closePalette();
    return;
  }
  if (event.key === "Escape") {
    // Close the top-most thing only.
    if (closePalette()) return;
    if (closeMenu({ restoreFocus: true })) return;
    if (!$("#modal").hidden) {
      closeModal();
      return;
    }
    if (closeContextMenu()) {
      $("#addContext").focus();
      return;
    }
    if (recorder) {
      stopRecording();
      return;
    }
    setInspector(false);
    setSidebar(false);
  }
});

window.addEventListener("needle:auth-required", () => showSignIn("Your session ended. Sign in again."));
if (/Mac|iPhone|iPad/.test(navigator.platform || "")) $$(".shortcut").forEach((node) => (node.textContent = "⌘K"));

function applyRole(session) {
  state.role = session.role || "owner";
  state.demo = Boolean(session.demo);
  state.voiceSeconds = session.voice_max_seconds || 15;
  $("#voiceButton").hidden = !session.voice;
  const visitor = state.role === "visitor";
  $("#app").classList.toggle("readonly", visitor);
  $("#demoBadge").hidden = !state.demo;
}

// The splash in index.html covers the first load; it fades once there is something to show.
function hideSplash() {
  const splash = $("#bootSplash");
  if (!splash) return;
  splash.classList.add("done");
  setTimeout(() => splash.remove(), reducedMotion() ? 0 : 420);
}

async function boot() {
  const session = await api.session();
  if (!session.authenticated) {
    hideSplash();
    showSignIn();
    return;
  }
  applyRole(session);
  await refreshShell();
  renderConversation();
  if (location.hash && location.hash !== "#ask") routeFromHash();
  hideSplash();
  moveRailPill(false);
}

listen(window, "resize", () => moveRailPill(false));
boot()
  .catch((err) => notify(err.message, { type: "error" }))
  .finally(hideSplash);
