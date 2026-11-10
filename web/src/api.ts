/** `node` is the peer machine the model runs on; absent for this one. */
/** `size` is the file's bytes and `ctx` the context it runs with; both 0 from older peers. */
export type ModelInfo = { id: string; node?: string | null; thinks: boolean; loaded: boolean; vision?: boolean; size?: number; ctx?: number };

/** A model's own name: a peer's models are listed as `name@node`. */
export const modelName = (m: ModelInfo) => (m.node ? m.id.slice(0, -(m.node.length + 1)) : m.id);

/** Where the app shows a conversation. */
export const convPath = (id: string) => `/c/${id}`;
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

/** One agent step as the server reports it; `label` is what to call it on screen. */
export type ToolCall = { id: string; name: string; label?: string; subject?: string; arguments: Record<string, unknown>; needs_approval: boolean };
export type ToolResult = { id: string; status: ToolStatus; output: string };

/** A file attached to a user message, as it is stored with it. */
export type AttachmentInfo = { id: string; name: string; chars: number; image?: boolean };
/** The file extensions the server accepts, by what it does with them. */
export type FileTypes = { documents: string[]; images: string[] };
/** An upload the server accepted: the workspace id goes back in the turn, the name is for display. */
export type UploadedFile = { id: string; name: string };

export type StoredMessage = {
  id: number;
  role: "user" | "assistant" | "tool";
  content: string;
  reasoning: string | null;
  skill: SkillInfo | null;
  /** An agent step; its output is the message's `content`. */
  tool: (Omit<ToolCall, "needs_approval"> & { status: ToolStatus; raw: string }) | null;
  refs: Ref[];
  attachments: AttachmentInfo[];
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

/** Who a node on the tailnet says it is. */
export type NodeInfo = { name: string; version: string; models: ModelInfo[] };

/** Runtime settings. The command line and the UI write the same shape; a flag wins over what was saved. */
export type Config = Partial<{
  searxng: string;
  skillsIndex: string;
  name: string;
  peers: string[];
  reasoning: Effort;
  trustTailnet: boolean;
}>;
export type Settings = {
  saved: Config;
  flags: Config;
  tailscale: { socket: boolean; status: TailStatus | null };
  discovered: NodeInfo[];
  system: {
    version: string;
    token: boolean;
    addr: string;
    name: string;
    autoName: string;
    reasoning: Effort;
    trustTailnet: boolean;
    workspace: string;
    models: ModelInfo[];
  };
};
/** This machine as tailscaled reports it; `null` when tailscaled did not answer. */
export type TailStatus = { running: boolean; state: string; host: string; dns: string; ips: string[]; url: string | null };

/** One event of a turn's stream, as the server's `turn::Out` names it. `error` and `stopped` arrive as thrown errors. */
export type TurnEvent =
  | { event: "skill"; data: SkillInfo }
  | { event: "thinking"; data: { text: string } }
  | { event: "answer"; data: { text: string } }
  | { event: "tool_call"; data: ToolCall }
  | { event: "tool_result"; data: ToolResult }
  | { event: "done"; data: { message: number; finish_reason: "stop" | "length"; tokens?: number } };

export class AuthError extends Error {}

/** The connection to the server ended mid-turn (it stopped, restarted or became unreachable); what it generated is saved. */
export class ConnectionLost extends Error {}

async function check(r: Response): Promise<Response> {
  if (r.status === 401) throw new AuthError("Esse token não abre este servidor.");
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error(body?.error?.message ?? `O servidor respondeu ${r.status}.`);
  }
  return r;
}

type Frame = { event: string; data: string };

/** Wraps one file in the field name every upload endpoint reads. */
function one(field: string, file: File): FormData {
  const form = new FormData();
  form.append(field, file);
  return form;
}

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
  /** Multipart calls set their own content-type; only the token rides along. */
  const upload = async <T>(path: string, form: FormData): Promise<T> => (await check(await fetch(path, { method: "POST", headers, body: form }))).json();

  return {
    /** Throws AuthError when the token is missing or wrong. */
    models: async (): Promise<ModelInfo[]> => (await get<{ data: ModelInfo[] }>("/v1/models")).data,
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
    /** A new skill; the server names it after `d.name`. */
    createSkill: (d: SkillDraft) => send<Skill>("POST", "/lair/v1/skills", d),
    saveSkill: (id: string, d: SkillDraft) => send<Skill>("PUT", `/lair/v1/skills/${encodeURIComponent(id)}`, d),
    toggleSkill: (id: string, enabled: boolean) => send<void>("PATCH", `/lair/v1/skills/${encodeURIComponent(id)}`, { enabled }),
    removeSkill: (id: string) => send<void>("DELETE", `/lair/v1/skills/${encodeURIComponent(id)}`),
    installSkill: (url: string) => send<Skill>("POST", "/lair/v1/skills/install", { url }),
    /** A skill in a `.zip`, read from the user's disk. */
    uploadSkill: (file: File) => upload<Skill>("/lair/v1/skills/upload", one("file", file)),
    skillsIndex: () => get<{ configured: boolean; skills: { entry: IndexEntry; installed: boolean }[] }>("/lair/v1/skills/index"),

    settings: () => get<Settings>("/lair/v1/settings"),
    /** Changes the named settings; `""` clears one, the others stay. */
    saveSettings: (change: Config) => send<void>("PATCH", "/lair/v1/settings", change),
    fileTypes: () => get<FileTypes>("/lair/v1/files"),
    /** Saves a file into the agent's workspace; the answer's `id` goes into the turn. */
    uploadFile: (file: File) => upload<UploadedFile>("/lair/v1/files", one("file", file)),
    /** An attachment as a blob URL: `<img>` can't send the token, so it is fetched here. Revoke when done. */
    fileUrl: async (id: string) => URL.createObjectURL(await (await check(await fetch(`/lair/v1/files/${encodeURIComponent(id)}`, { headers }))).blob()),

    /** Streams one turn. Aborting `signal` disconnects; the server stops and keeps what was generated. */
    async turn(
      id: string,
      input: { content: string; refs: string[]; model: string; effort?: Effort; agent: boolean; attachments: UploadedFile[] },
      signal: AbortSignal,
      on: (e: TurnEvent) => void,
    ): Promise<void> {
      // fetch and read reject with a TypeError when the connection itself fails; an abort is the caller's own doing.
      const lost = (e: unknown) => {
        throw e instanceof TypeError && !signal.aborted ? new ConnectionLost("a conexão caiu") : e;
      };
      const r = await check(await fetch(`${conv(id)}/messages`, { method: "POST", headers: json, body: JSON.stringify(input), signal }).catch(lost));
      const reader = r.body!.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read().catch(lost);
        if (done) return;
        let frames;
        [frames, buf] = takeFrames(buf + value);
        for (const { event, data } of frames) {
          const d = JSON.parse(data);
          if (event === "error") throw new Error(d.message);
          if (event === "stopped") throw new ConnectionLost("o servidor parou");
          on({ event, data: d } as TurnEvent);
        }
      }
    },
  };
}
