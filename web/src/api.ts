export async function api<T>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(`/api/v2${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(path.includes("fmu") ? 180000 : 30000),
  });
  const data = await response.json();
  if (!response.ok) {
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : data.detail?.map((e: { msg: string }) => e.msg).join("; ");
    throw new Error(detail || `Request failed (${response.status}).`);
  }
  return data as T;
}

export function downloadJson(name: string, value: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(value, null, 2)], { type: "application/json" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export type Work = <T>(fn: () => Promise<T>) => Promise<T | undefined>;
