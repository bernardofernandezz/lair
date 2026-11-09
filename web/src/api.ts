/** `node` is the peer machine the model runs on; absent for this one. */
export type ModelInfo = { id: string; node?: string; thinks: boolean; loaded: boolean };
/** `reasoning_effort` values the server understands. */
export type Effort = "none" | "low" | "high";
export type SkillInfo = { id: string; name: string };
export type Ref = { id: string; title: string };

export type Conversation = {
  id: string;
  title: string;
  model: string | null;
  effort: Effort | null;
  agent: boolean;
  created: number;
  updated: number;
};

export type ToolStatus = "ok" | "error" | "denied";

/** One agent step as the server reports it. */
export type ToolCall = { id: string; name: string; arguments: Record<string, unknown>; needs_approval: boolean };
export type ToolResult = { id: string; status: ToolStatus; output: string };

export type StoredMessage = {
  id: number;
  role: "user" | "assistant" | "tool";
  content: string;
  reasoning: string | null;
  skill: SkillInfo | null;
  tool: (ToolCall & ToolResult) | null;
  refs: Ref[];
  created: number;
};

export type Hit = { id: string; title: string; snippet: string };

export type Skill = {
  id: string;
  name: string;
  description: string;
  when: string[];
  prompt: string;
  origin: "builtin" | "extra" | "created" | "installed";
  source: string | null;
  enabled: boolean;
};

export type SkillDraft = Pick<Skill, "name" | "description" | "when" | "prompt">;
export type IndexEntry = { id: string; name: string; description: string; url: string };

export type TurnHandlers = {
  skill: (s: SkillInfo) => void;
  thinking: (text: string) => void;
  answer: (text: string) => void;
  toolCall: (c: ToolCall) => void;
  toolResult: (r: ToolResult) => void;
  done: (d: { message: number; finish_reason: string }) => void;
};

export class AuthError extends Error {}

/** The server stopped (restart, shutdown) during a turn; what it generated is saved. */
export class ServerStopped extends Error {}

async function check(r: Response): Promise<Response> {
  if (r.status === 401) throw new AuthError("Esse token não abre este servidor.");
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `O servidor respondeu ${r.status}.`);
  }
  return r;
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

export type Api = ReturnType<typeof client>;

/** Every server call, authenticated with `token` (empty when the server has none). */
export function client(token: string) {
  const headers: Record<string, string> = token ? { Authorization: `Bearer ${token}` } : {};
  const json = { ...headers, "content-type": "application/json" };
  const get = async <T>(path: string): Promise<T> => (await check(await fetch(path, { headers }))).json();
  const send = async <T>(method: string, path: string, body?: unknown): Promise<T> => {
    const r = await check(await fetch(path, { method, headers: json, body: body === undefined ? undefined : JSON.stringify(body) }));
    return r.status === 204 ? (undefined as T) : r.json();
  };
  const conv = (id: string) => `/lair/v1/conversations/${encodeURIComponent(id)}`;

  return {
    /** Throws AuthError when the token is missing or wrong. */
    models: async (): Promise<ModelInfo[]> =>
      (await get<{ data: ModelInfo[] }>("/v1/models")).data.map((m) => ({ id: m.id, node: m.node ?? undefined, thinks: !!m.thinks, loaded: !!m.loaded })),
    /**
     * Trades the token for the HttpOnly `lair_asset` cookie.
     * GLTF textures are loaded by `<img>`, which cannot send an Authorization header.
     */
    session: () => send<void>("POST", "/v1/session"),
    conversations: () => get<Conversation[]>("/lair/v1/conversations"),
    search: (q: string) => get<Hit[]>(`/lair/v1/conversations/search?q=${encodeURIComponent(q)}`),
    create: (c: Partial<Pick<Conversation, "model" | "effort" | "agent">>) => send<Conversation>("POST", "/lair/v1/conversations", c),
    read: (id: string) => get<{ conversation: Conversation; messages: StoredMessage[] }>(conv(id)),
    update: (id: string, c: Partial<Pick<Conversation, "title" | "model" | "effort" | "agent">>) =>
      send<Conversation>("PATCH", conv(id), c),
    remove: (id: string) => send<void>("DELETE", conv(id)),
    /** Answers a tool call that is waiting for the user. */
    approve: (callId: string, d: { allow: boolean; always?: boolean }) =>
      send<void>("POST", `/lair/v1/approvals/${encodeURIComponent(callId)}`, d),

    skills: () => get<Skill[]>("/lair/v1/skills"),
    saveSkill: (id: string, d: SkillDraft) => send<Skill>("PUT", `/lair/v1/skills/${encodeURIComponent(id)}`, d),
    toggleSkill: (id: string, enabled: boolean) => send<void>("PATCH", `/lair/v1/skills/${encodeURIComponent(id)}`, { enabled }),
    removeSkill: (id: string) => send<void>("DELETE", `/lair/v1/skills/${encodeURIComponent(id)}`),
    installSkill: (url: string) => send<Skill>("POST", "/lair/v1/skills/install", { url }),
    skillsIndex: () => get<{ configured: boolean; skills: { entry: IndexEntry; installed: boolean }[] }>("/lair/v1/skills/index"),

    /** Streams one turn. Aborting `signal` disconnects; the server stops and keeps what was generated. */
    async turn(
      id: string,
      input: { content: string; refs: string[]; model: string; effort: Effort; agent: boolean },
      signal: AbortSignal,
      on: TurnHandlers,
    ): Promise<void> {
      const r = await check(await fetch(`${conv(id)}/messages`, { method: "POST", headers: json, body: JSON.stringify(input), signal }));
      const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) return;
        let frames;
        [frames, buf] = takeFrames(buf + value);
        for (const { event, data } of frames) {
          const d = JSON.parse(data);
          if (event === "error") throw new Error(d.message);
          if (event === "stopped") throw new ServerStopped("o servidor parou");
          if (event === "skill") on.skill(d);
          else if (event === "thinking") on.thinking(d.text);
          else if (event === "answer") on.answer(d.text);
          else if (event === "tool_call") on.toolCall(d);
          else if (event === "tool_result") on.toolResult(d);
          else if (event === "done") on.done(d);
        }
      }
    },
  };
}
