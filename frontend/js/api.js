export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

export const errorMessage = (data, fallback = "Request failed") => {
  const detail = data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (detail && typeof detail === "object" && !Array.isArray(detail) && typeof detail.message === "string") return detail.message;
  if (Array.isArray(detail)) {
    const text = detail
      .map((item) => (typeof item === "string" ? item : item?.msg))
      .filter(Boolean)
      .join(" ");
    if (text) return text;
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
  documents: () => json("/api/documents"),
  document: (documentId) => json(`/api/documents/${id(documentId)}`),
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
  feedback: (messageId, rating) => json(`/api/messages/${id(messageId)}/feedback`, { method: "POST", body: JSON.stringify({ rating }) }),
  reset: () => json("/api/workspace/reset", { method: "POST", body: JSON.stringify({ confirm: "DELETE" }) }),
  chat: (body) =>
    request("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  upload: async (file) => {
    const body = new FormData();
    body.append("file", file);
    const response = await request("/api/upload", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new ApiError(errorMessage(data, "Upload failed"), response.status, data?.detail);
    return data;
  },
};
