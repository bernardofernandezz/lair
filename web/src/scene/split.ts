import { parseScene } from "./schema";

export type Segment =
  | { kind: "md"; text: string }
  | { kind: "scene"; source: string; closed: boolean };

type Fence = { char: "`" | "~"; len: number; info: string; lang: string };

/**
 * Splits an assistant message into Markdown and 3D scene blocks.
 *
 * ````scene` and ````3d` fences become scenes; ````json` fences are accepted as
 * a fallback when their body parses as a scene (small models sometimes ignore
 * the fence language). An unclosed scene fence stays pending while streaming.
 */
export function splitMessage(content: string): Segment[] {
  const lines = content.replace(/\r\n?/g, "\n").split("\n");
  const segments: Segment[] = [];
  let mdStart = 0;
  let i = 0;

  const flushMd = (endLine: number) => {
    if (endLine <= mdStart) return;
    const text = lines.slice(mdStart, endLine).join("\n");
    if (text) segments.push({ kind: "md", text });
  };

  while (i < lines.length) {
    const open = fenceInfo(lines[i]);
    if (!open) {
      i++;
      continue;
    }
    const isScene = open.lang === "scene" || open.lang === "3d";
    const close = findClose(lines, i + 1, open);

    if (!isScene) {
      if (close >= 0) {
        if (open.lang === "json") {
          const source = lines.slice(i + 1, close).join("\n");
          if (parseScene(source).ok) {
            flushMd(i);
            segments.push({ kind: "scene", source, closed: true });
            i = close + 1;
            mdStart = i;
            continue;
          }
        }
        i = close + 1;
        continue;
      }
      i = lines.length;
      continue;
    }

    flushMd(i);
    if (close >= 0) {
      segments.push({ kind: "scene", source: lines.slice(i + 1, close).join("\n"), closed: true });
      i = close + 1;
      mdStart = i;
    } else {
      segments.push({ kind: "scene", source: lines.slice(i + 1).join("\n"), closed: false });
      i = lines.length;
      mdStart = i;
      break;
    }
  }

  flushMd(lines.length);
  return segments;
}

function findClose(lines: string[], from: number, open: Fence): number {
  for (let i = from; i < lines.length; i++) {
    const f = fenceInfo(lines[i]);
    if (f && f.char === open.char && f.len >= open.len && f.info.trim() === "") return i;
  }
  return -1;
}

function fenceInfo(line: string): Fence | null {
  const m = /^ {0,3}(`{3,}|~{3,})(.*)$/.exec(line);
  if (!m) return null;
  const fence = m[1];
  const info = m[2];
  return {
    char: fence[0] as "`" | "~",
    len: fence.length,
    info,
    lang: (info.trim().split(/\s+/)[0] ?? "").toLowerCase(),
  };
}
