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

import { citedIn, linkCitations, parseSources, sourcesOf } from "./agent/sources";

const searchOutput = JSON.stringify([
  { n: 1, title: "Um", url: "https://www.a.com/x", snippet: "s" },
  { n: 2, title: "Dois", url: "https://b.org", snippet: "" },
  { n: 3, title: "Mau", url: "javascript:alert(1)", snippet: "" },
]);

test("parseSources keeps numbered http(s) results and ignores anything else", () => {
  expect(parseSources(searchOutput).map((s) => s.n)).toEqual([1, 2]);
  expect(parseSources("nenhum resultado")).toEqual([]);
  expect(parseSources("[não é json")).toEqual([]);
});

test("linkCitations links known sources only, outside code", () => {
  const known = sourcesOf([
    { kind: "tool", step: { id: "a", name: "web_search", label: "", subject: "", arguments: {}, needsApproval: false, status: "ok", output: searchOutput } },
  ]);
  expect(linkCitations("Fato [1]. Outro [1, 2]. Falso [9].", known)).toBe("Fato [1](#src-1). Outro [1](#src-1)[2](#src-2). Falso [9].");
  expect(linkCitations("Use `a[1]` e\n```\nb[2]\n```\nok [2]", known)).toBe("Use `a[1]` e\n```\nb[2]\n```\nok [2](#src-2)");
  expect(linkCitations("já é [1](https://x)", known)).toBe("já é [1](https://x)");
  expect(citedIn("b [2] a [1] de novo [2]", known)).toEqual([2, 1]);
  expect(linkCitations("[1]", new Map())).toBe("[1]");
});

import { bareName } from "./Controls";

test("bareName drops a trailing quantization only", () => {
  expect(bareName("Qwen3-4B-Q4_K_M")).toBe("Qwen3-4B");
  expect(bareName("gemma-3-4b-it-q8_0")).toBe("gemma-3-4b-it");
  expect(bareName("Qwen3-4B")).toBe("Qwen3-4B");
});
