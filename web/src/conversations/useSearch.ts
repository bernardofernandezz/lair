import { useEffect, useState } from "react";
import type { Api, Hit } from "../api";

/** Server-side conversation search for `q`, debounced; `null` while `q` is empty. */
export function useSearch(api: Api, q: string): Hit[] | null {
  const [hits, setHits] = useState<Hit[] | null>(null);
  useEffect(() => {
    if (!q.trim()) return setHits(null);
    let alive = true;
    const t = setTimeout(() => {
      api.search(q).then(
        (h) => alive && setHits(h),
        () => alive && setHits([]),
      );
    }, 200);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [api, q]);
  return hits;
}
