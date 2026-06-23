const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export function apiStream(path: string, onEvent: (event: string, data: string) => void) {
  const es = new EventSource(`${API_BASE}${path}`);
  es.addEventListener("progress", (e) => onEvent("progress", e.data));
  es.addEventListener("complete", (e) => { onEvent("complete", e.data); es.close(); });
  es.addEventListener("error", (e) => {
    if (es.readyState === EventSource.CLOSED) return;
    onEvent("error", "Connection lost");
    es.close();
  });
  return es;
}
