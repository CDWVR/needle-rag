export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

// FastAPI validation errors describe the schema ("String should match pattern '^[0-9a-f]…'").
// People see a sentence about what they tried instead; the field name says which one.
const validationMessage = (items) => {
  const fields = items.map((item) => (Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : "")).filter(Boolean);
  if (fields.some((field) => /_id$|^id$/.test(String(field)))) return "That link points to something that no longer exists.";
  if (fields.includes("title")) return "Give the conversation a name.";
  if (fields.includes("query")) return "Type a question first.";
  return "Some of what you entered isn't valid. Check the highlighted values and try again.";
};

export const errorMessage = (data, fallback = "Something went wrong. Try again.") => {
  const detail = data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail) && typeof detail.message === "string") return detail.message;
  if (Array.isArray(detail) && detail.length) {
    if (detail.every((item) => typeof item === "string")) return detail.join(" ");
    return validationMessage(detail);
  }
  return fallback;
};

// Every request carries this header. A page on another site cannot set it without a CORS
// preflight the server never grants, so the server rejects cross-site writes that lack it.
const CSRF = { "X-Needle-CSRF": "1" };
const id = (value) => encodeURIComponent(String(value ?? ""));

// The app listens for this and shows the sign-in screen.
const authRequired = () => window.dispatchEvent(new CustomEvent("needle:auth-required"));

export const request = async (url, options = {}) => {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: { ...CSRF, ...(options.headers || {}) },
  });
  if (response.status === 401 && !url.startsWith("/api/auth/")) authRequired();
  return response;
};

const json = async (url, options = {}) => {
  const response = await request(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new ApiError(errorMessage(data), response.status, data?.detail);
  return data;
};

export const api = {
  session: () => json("/api/auth/session"),
  login: (token) => json("/api/auth/login", { method: "POST", body: JSON.stringify({ token }) }),
  logout: () => json("/api/auth/logout", { method: "POST", body: "{}" }),
  settings: () => json("/api/settings"),
  saveSettings: (body) => json("/api/settings", { method: "PUT", body: JSON.stringify(body) }),
  conversations: (q = "") => json(`/api/conversations?q=${encodeURIComponent(q)}`),
  createConversation: () => json("/api/conversations", { method: "POST", body: "{}" }),
  conversation: (conversationId) => json(`/api/conversations/${id(conversationId)}`),
  renameConversation: (conversationId, title) =>
    json(`/api/conversations/${id(conversationId)}`, { method: "PATCH", body: JSON.stringify({ title }) }),
  deleteConversation: (conversationId) => json(`/api/conversations/${id(conversationId)}`, { method: "DELETE" }),
  documents: () => json("/api/documents"),
  document: (documentId, { offset = 0, limit = 40, focus = "" } = {}) =>
    json(`/api/documents/${id(documentId)}?offset=${offset}&limit=${limit}${focus ? `&focus=${id(focus)}` : ""}`),
  updateDocument: (documentId, body) => json(`/api/documents/${id(documentId)}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteDocument: (documentId) => json(`/api/documents/${id(documentId)}`, { method: "DELETE" }),
  documentFile: (documentId) => request(`/api/documents/${id(documentId)}/file`),
  job: (jobId) => json(`/api/jobs/${id(jobId)}`),
  index: () => json("/api/index"),
  refreshIndex: (override = false) =>
    json(`/api/index/refresh${override ? "?publish_override=true" : ""}`, { method: "POST", body: "{}" }),
  indexVersions: () => json("/api/index/versions"),
  evalLatest: () => json("/api/eval/latest"),
  rollbackIndex: () => json("/api/index/rollback", { method: "POST", body: "{}" }),
  reconcileIndex: () => json("/api/index/reconcile", { method: "POST", body: "{}" }),
  analytics: (days) => json(`/api/analytics?days=${id(days)}`),
  analyticsExport: (days) => request(`/api/analytics/export?days=${id(days)}`),
  feedback: (messageId, rating, reason = null) =>
    json(`/api/messages/${id(messageId)}/feedback`, { method: "POST", body: JSON.stringify(reason ? { rating, reason } : { rating }) }),
  reset: () => json("/api/workspace/reset", { method: "POST", body: JSON.stringify({ confirm: "DELETE" }) }),
  chat: (body, signal) =>
    request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal }),
  transcribe: async (blob, seconds) => {
    const body = new FormData();
    body.append("file", blob, "voice");
    body.append("seconds", String(Math.round(seconds * 10) / 10));
    body.append("language", (navigator.language || "").split("-")[0]);
    const response = await request("/api/transcribe", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(errorMessage(data, "Voice input failed"), response.status, data?.detail);
    return data;
  },
  upload: async (file) => {
    const body = new FormData();
    body.append("file", file);
    const response = await request("/api/upload", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(errorMessage(data, "Upload failed"), response.status, data?.detail);
    return data;
  },
};
