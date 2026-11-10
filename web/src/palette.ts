/** The Pearl's thin-film colours. The 3D sphere, the favicon and the page's accents all take them from here. */
export const PEARL = { teal: "#40b8b0", violet: "#7a5cd6", gold: "#e3b857", red: "#d05a5a" } as const;

/** A colour as a GLSL `vec3` literal. */
export function vec3(hex: string): string {
  const [r, g, b] = [1, 3, 5].map((i) => (parseInt(hex.slice(i, i + 2), 16) / 255).toFixed(3));
  return `vec3(${r}, ${g}, ${b})`;
}

/** Exposes the palette to the stylesheet as `--teal`, `--violet`, `--gold`, `--red`. */
export function paintTokens(root: HTMLElement) {
  for (const [name, hex] of Object.entries(PEARL)) root.style.setProperty(`--${name}`, hex);
}
