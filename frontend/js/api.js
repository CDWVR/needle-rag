export const errorMessage = (data, fallback = "Request failed") => {
  const detail = data?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const text = detail
      .map((item) => (typeof item === "string" ? item : item?.msg))
      .filter(Boolean)
      .join(" ");
    if (text) return text;
  }
  return fallback;
};

const json = async (url, options = {}) => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(data));
  return data;
};

export const api = {
  settings: () => json("/api/settings"),
  saveSettings: (body) => json("/api/settings", { method: "PUT", body: JSON.stringify(body) }),
  conversations: (q = "") => json(`/api/conversations?q=${encodeURIComponent(q)}`),
  createConversation: () => json("/api/conversations", { method: "POST", body: "{}" }),
  conversation: (id) => json(`/api/conversations/${id}`),
  documents: () => json("/api/documents"),
  document: (id) => json(`/api/documents/${id}`),
  updateDocument: (id, body) => json(`/api/documents/${id}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteDocument: (id) => json(`/api/documents/${id}`, { method: "DELETE" }),
  index: () => json("/api/index"),
  refreshIndex: () => json("/api/index/refresh", { method: "POST", body: "{}" }),
  rollbackIndex: () => json("/api/index/rollback", { method: "POST", body: "{}" }),
  reconcileIndex: () => json("/api/index/reconcile", { method: "POST", body: "{}" }),
  analytics: (days) => json(`/api/analytics?days=${days}`),
  members: () => json("/api/members"),
  invite: (body) => json("/api/members", { method: "POST", body: JSON.stringify(body) }),
  integrations: () => json("/api/integrations"),
  feedback: (id, rating) => json(`/api/messages/${id}/feedback`, { method: "POST", body: JSON.stringify({ rating }) }),
  reset: () => json("/api/workspace/reset", { method: "POST", body: JSON.stringify({ confirm: "DELETE" }) }),
  upload: async (file) => {
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/api/upload", { method: "POST", body });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(data, "Upload failed"));
    return data;
  },
};
