import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { parseScene } from "./scene/schema";
import { splitMessage } from "./scene/split";

const scene = `{"nodes":[{"geometry":{"kind":"sphere","radius":1}}]}`;

test("splitMessage keeps markdown around a scene fence", () => {
  const segments = splitMessage(`Olha:\n\n\`\`\`scene\n${scene}\n\`\`\`\n\nFim.`);
  expect(segments.map((s) => s.kind)).toEqual(["md", "scene", "md"]);
  expect(segments[1]).toMatchObject({ kind: "scene", source: scene, closed: true });
});

test("splitMessage accepts ```3d and an unclosed scene while streaming", () => {
  expect(splitMessage(`\`\`\`3d\n${scene}\n\`\`\``)[0]).toMatchObject({ kind: "scene", closed: true });
  const [open] = splitMessage(`\`\`\`scene\n${scene}`);
  expect(open).toMatchObject({ kind: "scene", closed: false });
});

test("splitMessage treats a ```json fence as a scene only when it parses", () => {
  expect(splitMessage(`\`\`\`json\n${scene}\n\`\`\``)[0]).toMatchObject({ kind: "scene" });
  const plain = splitMessage("```json\n{\"a\":1}\n```");
  expect(plain).toEqual([{ kind: "md", text: "```json\n{\"a\":1}\n```" }]);
});

test("parseScene fills defaults and clamps numbers", () => {
  const result = parseScene(`{"nodes":[{"geometry":{"kind":"box","size":[2,1,2]},"material":{"metalness":5}}]}`);
  expect(result.ok).toBe(true);
  if (!result.ok) return;
  expect(result.scene.grid).toBe(false);
  expect(result.scene.lights).toEqual([]);
  expect(result.scene.nodes[0].material.color).toBe("#8a8f9e");
  expect(result.scene.nodes[0].position).toEqual([0, 0, 0]);
  expect(result.scene.nodes[0].material.metalness).toBe(1);
  expect(result.scene.nodes[0].material.roughness).toBe(0.6);
});

test("parseScene accepts polyhedra and points", () => {
  const result = parseScene(
    `{
      nodes: [
        { geometry: { kind: "polyhedron", vertices: [[0,0,0],[1,0,0],[0,1,0]], faces: [[0,1,2]] } },
        { geometry: { kind: "points", positions: [[0,0,0],[1,1,1]], colors: ["#ffffff", "#ffffff"], size: 0.1 } }
      ]
    }`,
  );
  expect(result.ok).toBe(true);
  if (!result.ok) return;
  expect(result.scene.nodes[0].geometry).toMatchObject({ kind: "polyhedron" });
  expect(result.scene.nodes[1].geometry).toMatchObject({ kind: "points" });
});

test("parseScene requires one color per point", () => {
  const result = parseScene(
    `{"nodes":[{"geometry":{"kind":"points","positions":[[0,0,0],[1,1,1]],"colors":["#ffffff"]}}]}`,
  );
  expect(result.ok).toBe(false);
  if (!result.ok) expect(result.error).toContain("colors");
});

test("parseScene rejects unsafe asset ids", () => {
  for (const id of ["../secret.glb", "/etc/passwd.glb", "a/../b.glb", ".hidden.glb"]) {
    const result = parseScene(`{"nodes":[{"asset":"${id}"}]}`);
    expect(result.ok).toBe(false);
  }
  expect(parseScene(`{"nodes":[{"asset":"models/robot.glb"}]}`).ok).toBe(true);
});

test("parseScene rejects oversized and malformed scenes", () => {
  expect(parseScene(`{"nodes":[]}`).ok).toBe(false);
  expect(parseScene(`{"nodes":[{"geometry":{"kind":"nope"}}]}`).ok).toBe(false);
  expect(parseScene(`{"nodes":[{"geometry":{"kind":"box"}}]}`).ok).toBe(true);
  expect(parseScene("[]").ok).toBe(false);
});

test("the example in the 3D skill's prompt is a scene this schema accepts", () => {
  const prompt = readFileSync(new URL("../../skills/3d/prompt.md", import.meta.url), "utf8");
  const example = prompt.slice(prompt.indexOf("## Exemplo"));
  const bodies = [...example.matchAll(/```scene\n([\s\S]*?)```/g)].map((m) => m[1]);
  expect(bodies.length).toBeGreaterThanOrEqual(3);
  for (const body of bodies) {
    JSON.parse(body); // the prompt promises strict JSON, not JSON5
    const result = parseScene(body);
    expect(result.ok ? "" : result.error).toBe("");
  }
});

test("repeat copies a node in steps around the parent's origin", () => {
  const r = parseScene(
    `{"nodes":[{"geometry":{"kind":"box","size":[1,0.1,0.3]},"position":[1,0,0],"repeat":{"count":3,"offset":[0,0.2,0],"rotation":[0,0.5,0],"scale":[2,1,1]}}]}`,
  );
  expect(r.ok).toBe(true);
  if (!r.ok) return;
  expect(r.scene.nodes).toHaveLength(3);
  expect(r.scene.nodes[2].position).toEqual([0, 0.4, 0]);
  expect(r.scene.nodes[2].rotation).toEqual([0, 1, 0]);
  expect(r.scene.nodes[2].scale).toEqual([4, 1, 1]);
  expect(r.scene.nodes[2].children[0].position).toEqual([1, 0, 0]);
});

test("repeat counts its copies toward the node limit", () => {
  const child = `{"geometry":{"kind":"sphere","radius":0.1}}`;
  const many = `{"nodes":[{"children":[${child},${child},${child}],"repeat":{"count":100}}]}`;
  const r = parseScene(many);
  expect(r.ok).toBe(false);
  if (!r.ok) expect(r.error).toContain("repeats");
  const clamped = parseScene(`{"nodes":[{"geometry":{"kind":"box"},"repeat":{"count":0}}]}`);
  expect(clamped.ok && clamped.scene.nodes.length).toBe(1);
  expect(parseScene(`{"nodes":[{"geometry":{"kind":"box"},"repeat":{}}]}`).ok).toBe(false);
});
