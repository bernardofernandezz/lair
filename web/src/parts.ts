import type { Ref, SkillInfo, StoredMessage, ToolCall, ToolResult, ToolStatus } from "./api";

export type ToolStep = {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  needsApproval: boolean;
  /** `pending` waits for the user; `running` is executing. */
  status: "pending" | "running" | ToolStatus;
  output?: string;
};

/** An assistant reply, in the order things happened: text, a tool step, more text… */
export type Part = { kind: "text"; text: string } | { kind: "tool"; step: ToolStep };

export type Turn =
  | { role: "user"; content: string; refs?: Ref[] }
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

export function addCall(t: AssistantTurn, c: ToolCall) {
  t.parts.push({
    kind: "tool",
    step: { id: c.id, name: c.name, arguments: c.arguments ?? {}, needsApproval: c.needs_approval, status: c.needs_approval ? "pending" : "running" },
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
      out.push({ role: "user", content: m.content, refs: m.refs });
      continue;
    }
    if (!reply) {
      reply = emptyReply();
      out.push(reply);
    }
    if (m.reasoning) reply.reasoning = (reply.reasoning ? reply.reasoning + "\n\n" : "") + m.reasoning;
    if (m.skill) reply.skill = m.skill;
    if (m.role === "tool" && m.tool) {
      addCall(reply, { ...m.tool, needs_approval: false });
      setResult(reply, m.tool);
    } else if (m.content) {
      addText(reply, (reply.parts.at(-1)?.kind === "text" ? "\n\n" : "") + m.content);
    }
  }
  return out;
}
