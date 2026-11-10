import type { Part } from "../parts";

/** One web search result as `web_search` numbers it; `n` is what the model cites as `[n]`. */
export type Source = { n: number; title: string; url: string; snippet: string };

/** The results in a `web_search` output, or none. URLs come from the web: only http(s) ones are kept. */
export function parseSources(output?: string): Source[] {
  if (!output?.trimStart().startsWith("[")) return [];
  try {
    const list: unknown = JSON.parse(output);
    if (!Array.isArray(list)) return [];
    return list.filter(
      (s): s is Source => typeof s?.n === "number" && typeof s.url === "string" && /^https?:\/\//i.test(s.url),
    ).map((s) => ({ n: s.n, url: s.url, title: String(s.title ?? s.url), snippet: String(s.snippet ?? "") }));
  } catch {
    return [];
  }
}

/** Every result the reply's searches returned, by number. */
export function sourcesOf(parts: Part[]): Map<number, Source> {
  const out = new Map<number, Source>();
  for (const p of parts) {
    if (p.kind === "tool" && p.step.name === "web_search" && p.step.status === "ok") {
      for (const s of parseSources(p.step.output)) out.set(s.n, s);
    }
  }
  return out;
}

export const domain = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

const CODE = /(```[\s\S]*?(?:```|$)|`[^`\n]*`)/;
const CITE = /\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\](?!\()/g;

/**
 * Turns `[n]` and `[n, m]` citations of known sources into `#src-n` links the answer renders as
 * badges. Code is left alone, and so are numbers no search returned.
 */
export function linkCitations(md: string, known: Map<number, Source>): string {
  if (known.size === 0) return md;
  return md
    .split(CODE)
    .map((piece, i) =>
      i % 2
        ? piece
        : piece.replace(CITE, (whole, list: string) => {
            const ns = list.split(",").map((n) => Number(n.trim()));
            return ns.every((n) => known.has(n)) ? ns.map((n) => `[${n}](#src-${n})`).join("") : whole;
          }),
    )
    .join("");
}

/** The source numbers an answer cites, in order of first citation. */
export function citedIn(md: string, known: Map<number, Source>): number[] {
  const out = new Set<number>();
  for (const m of linkCitations(md, known).matchAll(/\(#src-(\d+)\)/g)) out.add(Number(m[1]));
  return [...out];
}
