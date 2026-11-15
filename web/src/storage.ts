// Per-browser preferences. Storage can be unavailable (private mode, blocked site data); the UI works without it.

type Key = "token" | "model" | "effort" | "sidebar";

export function load(key: Key): string {
  try {
    return localStorage.getItem(`lair.${key}`) ?? "";
  } catch {
    return "";
  }
}

export function save(key: Key, value: string) {
  try {
    if (value) localStorage.setItem(`lair.${key}`, value);
    else localStorage.removeItem(`lair.${key}`);
  } catch {}
}
