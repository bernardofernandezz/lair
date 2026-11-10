import { expect, test } from "bun:test";
import { takeFrames } from "./api";

test("takeFrames keeps a partial frame for the next read and reads event names", () => {
  const [frames, rest] = takeFrames('event: lair.skill\ndata: {"id":"3d"}\n\ndata: {"a":1}\n\ndata: [DO');
  expect(frames).toEqual([
    { event: "lair.skill", data: '{"id":"3d"}' },
    { event: "message", data: '{"a":1}' },
  ]);
  const [more] = takeFrames(rest + "NE]\n\n");
  expect(more).toEqual([{ event: "message", data: "[DONE]" }]);
});

import { mentionAt } from "./conversations/mention";

test("mentionAt finds the @query being typed before the caret", () => {
  expect(mentionAt("veja @pão", 9)).toEqual({ start: 5, query: "pão" });
  expect(mentionAt("@", 1)).toEqual({ start: 0, query: "" });
  expect(mentionAt("email a@b.com", 13)).toBeNull();
  expect(mentionAt("veja @pão e mais", 16)).toBeNull();
  expect(mentionAt("veja @pão e mais", 9)).toEqual({ start: 5, query: "pão" });
});

import { fromStored } from "./parts";
import type { StoredMessage } from "./api";

test("fromStored groups the assistant and tool messages after a question into one reply", () => {
  const m = (role: StoredMessage["role"], content: string, extra: Partial<StoredMessage> = {}): StoredMessage => ({
    id: 0, role, content, reasoning: null, skill: null, tool: null, refs: [], attachments: [], created: 0, ...extra,
  });
  const turns = fromStored([
    m("user", "liste"),
    m("assistant", "Vou olhar.", { reasoning: "plano" }),
    m("tool", "a.md", { tool: { id: "t1", name: "list_dir", arguments: { path: "." }, status: "ok", raw: "[list_dir(path='.')]" } }),
    m("assistant", "Há a.md."),
    m("user", "obrigado"),
  ]);
  expect(turns.map((t) => t.role)).toEqual(["user", "assistant", "user"]);
  const reply = turns[1];
  if (reply.role !== "assistant") throw new Error("expected a reply");
  expect(reply.reasoning).toBe("plano");
  expect(reply.parts.map((p) => p.kind)).toEqual(["text", "tool", "text"]);
  expect(reply.parts[1]).toMatchObject({ kind: "tool", step: { name: "list_dir", status: "ok", output: "a.md" } });
});

import { groupByNode, quantOf } from "./Controls";

test("quantOf reads the quantization from GGUF file names", () => {
  expect(quantOf("Qwen3-4B-Q4_K_M")).toBe("Q4_K_M");
  expect(quantOf("gemma-3-4b-it-q8_0")).toBe("Q8_0");
  expect(quantOf("Llama-3.2-3B-Instruct-IQ4_XS")).toBe("IQ4_XS");
  expect(quantOf("model-BF16")).toBe("BF16");
  expect(quantOf("Qwen3-4B")).toBeNull();
});

test("groupByNode puts this machine first and keeps each node's order", () => {
  const m = (id: string, node?: string) => ({ id, node, thinks: false, loaded: false });
  const groups = groupByNode([m("a@x", "x"), m("b"), m("c@y", "y"), m("d")]);
  expect(groups.map(([n, ms]) => [n, ms.map((x) => x.id)])).toEqual([
    [null, ["b", "d"]],
    ["x", ["a@x"]],
    ["y", ["c@y"]],
  ]);
  expect(groupByNode([m("a@x", "x")]).map(([n]) => n)).toEqual(["x"]);
});

import { bareName } from "./Controls";

test("bareName drops a trailing quantization only", () => {
  expect(bareName("Qwen3-4B-Q4_K_M")).toBe("Qwen3-4B");
  expect(bareName("gemma-3-4b-it-q8_0")).toBe("gemma-3-4b-it");
  expect(bareName("Qwen3-4B")).toBe("Qwen3-4B");
});
