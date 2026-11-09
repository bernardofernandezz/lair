export type Msg = { role: "user" | "assistant"; content: string };
export type ModelInfo = { id: string; thinks: boolean; loaded: boolean };
/** `reasoning_effort` values the server understands. */
export type Effort = "none" | "low" | "high";
export type Delta = { reasoning?: string; content?: string };
/** A skill the server activated for this reply. */
export type SkillInfo = { id: string; name: string };

export class AuthError extends Error {}

function auth(token: string): Record<string, string> {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function check(r: Response): Promise<Response> {
  if (r.status === 401) throw new AuthError("Esse token não abre este servidor.");
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `O servidor respondeu ${r.status}.`);
  }
  return r;
}

/** Throws AuthError when the token is missing or wrong. */
export async function listModels(token: string): Promise<ModelInfo[]> {
  const r = await check(await fetch("/v1/models", { headers: auth(token) }));
  const body = await r.json();
  return body.data.map((m: ModelInfo) => ({ id: m.id, thinks: !!m.thinks, loaded: !!m.loaded }));
}

/**
 * Trades the Bearer token for the HttpOnly `lair_asset` cookie.
 * GLTF textures are loaded by `<img>`, which cannot send an Authorization header.
 */
export async function createSession(token: string): Promise<void> {
  await check(await fetch("/v1/session", { method: "POST", headers: auth(token) }));
}

export type Frame = { event: string; data: string };

/** Splits complete SSE frames off the front of `buf`; returns them and the unfinished rest. */
export function takeFrames(buf: string): [Frame[], string] {
  const parts = buf.split("\n\n");
  const rest = parts.pop()!;
  const frames = parts.map((raw) => {
    let event = "message";
    const data: string[] = [];
    for (const line of raw.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    return { event, data: data.join("\n") };
  });
  return [frames.filter((f) => f.data), rest];
}

/** Streams a reply. Aborting `signal` disconnects, which makes the server stop generating. */
export async function streamChat(
  req: { model: string; effort: Effort; messages: Msg[] },
  token: string,
  signal: AbortSignal,
  onDelta: (d: Delta) => void,
  onSkill: (s: SkillInfo) => void,
): Promise<void> {
  const r = await check(
    await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json", ...auth(token) },
      body: JSON.stringify({ model: req.model, reasoning_effort: req.effort, messages: req.messages, stream: true }),
      signal,
    }),
  );
  const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    let frames;
    [frames, buf] = takeFrames(buf + value);
    for (const frame of frames) {
      if (frame.event === "lair.skill") {
        onSkill(JSON.parse(frame.data));
        continue;
      }
      if (frame.data === "[DONE]") return;
      const chunk = JSON.parse(frame.data);
      if (chunk.error) throw new Error(chunk.error.message);
      const d = chunk.choices?.[0]?.delta ?? {};
      if (d.reasoning_content || d.content) onDelta({ reasoning: d.reasoning_content, content: d.content });
    }
  }
}
