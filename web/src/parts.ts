import type { Ref, SkillInfo, StoredMessage, ToolCall, ToolResult, ToolStatus } from "./api";

export type ToolStep = {
  id: string;
  name: string;
  label: string;
  /** The argument that says what the step is about: a path, a command, a query… */
  subject: string;
  arguments: Record<string, unknown>;
  needsApproval: boolean;
  /** `pending` waits for the user; `running` is executing. */
  status: "pending" | "running" | ToolStatus;
  output?: string;
};

/** An assistant reply, in the order things happened: text, a tool step, more text… */
export type Part = { kind: "text"; text: string } | { kind: "tool"; step: ToolStep };

/** An attachment shown on a user turn: `src` while the local file is at hand, `id` to fetch it later. */
export type Attached = { name: string; image?: boolean; id?: string; src?: string };

export type Turn =
  | { role: "user"; content: string; refs?: Ref[]; attachments?: Attached[] }
  | { role: "assistant"; parts: Part[]; reasoning?: string; thinkMs?: number; skill?: SkillInfo | null };

export type AssistantTurn = Extract<Turn, { role: "assistant" }>;

export function emptyReply(): AssistantTurn {
  return { role: "assistant", parts: [], reasoning: "" };
}

export function addText(t: AssistantTurn, text: string) {
  const last = t.parts.at(-1);
  if (last?.kind === "text") last.text += text;
  else t.parts.push({ kind: "text", text });
}

/** A tool step joins the reply; `waits` when it needs the user's approval before it runs. */
export function addCall(t: AssistantTurn, c: Omit<ToolCall, "needs_approval">, waits = false) {
  t.parts.push({
    kind: "tool",
    step: { id: c.id, name: c.name, label: c.label || c.name, subject: c.subject ?? "", arguments: c.arguments ?? {}, needsApproval: waits, status: waits ? "pending" : "running" },
  });
}

export function setResult(t: AssistantTurn, r: ToolResult) {
  for (const p of t.parts) if (p.kind === "tool" && p.step.id === r.id) Object.assign(p.step, { status: r.status, output: r.output });
}

export function hasText(t: AssistantTurn): boolean {
  return t.parts.some((p) => p.kind === "text" && p.text.trim() !== "");
}

/** A deep enough copy for React to see a new turn while it streams. */
export function snapshot(t: AssistantTurn): AssistantTurn {
  return { ...t, parts: t.parts.map((p) => (p.kind === "text" ? { ...p } : { kind: "tool", step: { ...p.step } })) };
}

/** Saved messages as turns: the assistant and tool messages after a question form one reply. */
export function fromStored(messages: StoredMessage[]): Turn[] {
  const out: Turn[] = [];
  let reply: AssistantTurn | null = null;
  for (const m of messages) {
    if (m.role === "user") {
      reply = null;
      out.push({ role: "user", content: m.content, refs: m.refs, attachments: m.attachments.map((a) => ({ name: a.name, image: !!a.image, id: a.id })) });
      continue;
    }
    if (!reply) {
      reply = emptyReply();
      out.push(reply);
    }
    if (m.reasoning) reply.reasoning = (reply.reasoning ? reply.reasoning + "\n\n" : "") + m.reasoning;
    if (m.skill) reply.skill = m.skill;
    if (m.role === "tool" && m.tool) {
      addCall(reply, m.tool);
      setResult(reply, { id: m.tool.id, status: m.tool.status, output: m.content });
    } else if (m.content) {
      addText(reply, (reply.parts.at(-1)?.kind === "text" ? "\n\n" : "") + m.content);
    }
  }
  return out;
}
