/** The Pearl as the favicon: a small obsidian sphere whose rim takes the current turn state.
 *
 * Chat writes state into the shared signal ref; this module watches it cheaply and redraws the
 * 32×32 icon only when the discrete state actually changes. */
import type { OrbSignal } from "./Orb";
import { PEARL } from "./palette";

type OrbPic = "idle" | "thinking" | "tool" | "waiting" | "error";

const RIM: Record<OrbPic, string> = {
  idle: PEARL.teal,
  thinking: PEARL.violet,
  // A blue apart from the film, so a running tool reads differently from plain thinking at 16 px.
  tool: "#60a5e0",
  waiting: PEARL.gold,
  error: PEARL.red,
};

let last: OrbPic | null = null;
let link: HTMLLinkElement | null = null;

/** What the shadowed signal means for an icon; bad news outranks an approval, which outranks work. */
function orbPic(sig: OrbSignal): OrbPic {
  if (sig.error) return "error";
  if (sig.waiting) return "waiting";
  if (sig.tool) return "tool";
  if (sig.thinking) return "thinking";
  return "idle";
}

function draw(state: OrbPic): string {
  const c = document.createElement("canvas");
  c.width = c.height = 32;
  const g = c.getContext("2d")!;
  const rim = RIM[state];
  const disc = g.createRadialGradient(13, 11, 2, 16, 16, 15);
  disc.addColorStop(0, "#2c2834");
  disc.addColorStop(0.72, "#161420");
  disc.addColorStop(1, rim);
  g.fillStyle = disc;
  g.beginPath();
  g.arc(16, 16, 13, 0, 7);
  g.fill();
  const glow = g.createRadialGradient(22, 22, 0, 22, 22, 10);
  glow.addColorStop(0, `${rim}66`);
  glow.addColorStop(1, "#0000");
  g.globalCompositeOperation = "screen";
  g.fillStyle = glow;
  g.beginPath();
  g.arc(22, 22, 10, Math.PI * 0.4, Math.PI, false);
  g.fill();
  return c.toDataURL();
}

function show(state: OrbPic) {
  if (!link) {
    link = document.querySelector<HTMLLinkElement>('link[rel="icon"]') ?? null;
    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      document.head.append(link);
    }
  }
  if (last === state) return;
  last = state;
  link.href = draw(state);
}

/** Watches the shared signal until the page goes away; redraws only on state changes. */
export function watchFavicon(signal: () => OrbSignal | null) {
  const tick = () => {
    const sig = signal();
    if (sig) show(orbPic(sig));
  };
  tick();
  const timer = setInterval(tick, 600);
  return () => clearInterval(timer);
}
