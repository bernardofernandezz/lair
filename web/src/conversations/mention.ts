import { useMemo } from "react";
import type { Api, Conversation, Ref } from "../api";
import { useSearch } from "./useSearch";

/** The `@query` being typed right before the caret, if any. */
export function mentionAt(text: string, caret: number): { start: number; query: string } | null {
  const m = /(^|\s)@([^\s@]*)$/.exec(text.slice(0, caret));
  return m ? { start: caret - m[2].length - 1, query: m[2] } : null;
}

/** Conversations to offer for `@query`: title matches first, then content matches from the server. */
export function useMentionOptions(api: Api, conversations: Conversation[], query: string | null, exclude: string[]): Ref[] {
  const hits = useSearch(api, query && query.length >= 2 ? query : "");
  return useMemo(() => {
    if (query === null) return [];
    const q = query.toLowerCase();
    const byTitle = conversations.filter((c) => c.title.toLowerCase().includes(q));
    const seen = new Set(exclude);
    const out: Ref[] = [];
    for (const c of [...byTitle, ...(hits ?? [])]) {
      if (seen.has(c.id) || !c.title) continue;
      seen.add(c.id);
      out.push({ id: c.id, title: c.title });
    }
    return out.slice(0, 6);
  }, [conversations, hits, query, exclude]);
}
