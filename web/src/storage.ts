// Per-browser preferences. Storage can be unavailable (private mode, blocked site data); the UI works without it.

export function load(key: string): string {
  try {
    return localStorage.getItem(`lair.${key}`) ?? "";
  } catch {
    return "";
  }
}

export function save(key: string, value: string) {
  try {
    if (value) localStorage.setItem(`lair.${key}`, value);
    else localStorage.removeItem(`lair.${key}`);
  } catch {}
}
