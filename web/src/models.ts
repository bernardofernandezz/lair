// How models are described on screen: their names, sizes, speeds and what they can do.
import { AUTO, modelName, type Effort, type ModelInfo } from "./api";
import { lang, pick } from "./i18n";

const T = pick({
  en: {
    modes: [
      { value: "none" as Effort, label: "Off", hint: "Answers right away, no thinking" },
      { value: "low" as Effort, label: "Light", hint: "Thinks a little before answering" },
      { value: "high" as Effort, label: "Deep", hint: "Thinks as long as it needs" },
    ],
    auto: "Auto",
    speed: (t: string) => `${t} tok/s`,
    measured: "Measured on this machine",
    estimated: "Estimated from other models measured on this machine",
  },
  pt: {
    modes: [
      { value: "none" as Effort, label: "Desligado", hint: "Responde direto, sem pensar" },
      { value: "low" as Effort, label: "Leve", hint: "Pensa um pouco antes de responder" },
      { value: "high" as Effort, label: "Profundo", hint: "Pensa o quanto precisar" },
    ],
    auto: "Auto",
    speed: (t: string) => `${t} tok/s`,
    measured: "Medido nesta máquina",
    estimated: "Estimado a partir de outros modelos medidos nesta máquina",
  },
});

export const MODES: { value: Effort; label: string; hint: string }[] = T.modes;

export const isEffort = (v: string): v is Effort => MODES.some((m) => m.value === v);

// shortcut: read from the file name; read general.file_type from the GGUF if names outside the pattern show up.
const QUANT = /(?:^|[-_.])(I?Q\d(?:_[KS0-9])?(?:_[A-Z0-9]+)?|BF16|F16|F32)(?=$|[-_.])/i;

/** The quantization in a GGUF file name, like `Q4_K_M`, or null. */
export const quantOf = (id: string) => id.match(QUANT)?.[1].toUpperCase() ?? null;

/** Tokens per second as a tag's text: `~23 tok/s` when estimated, `23 tok/s` when measured; null when unknown. */
export const speedLabel = (tps: number | undefined, measured?: boolean) =>
  tps && tps > 0 ? `${measured ? "" : "~"}${T.speed(num(tps, tps < 10 ? 1 : 0))}` : null;
export const speedHint = (measured?: boolean) => (measured ? T.measured : T.estimated);

/** What a model is called on screen: its own name without the quantization, or "Auto". */
export const displayName = (m: ModelInfo) => (m.id === AUTO ? T.auto : bareName(modelName(m)));

/** The name without its quantization, which is shown apart as a tag. */
export const bareName = (name: string) => {
  const q = quantOf(name);
  return q ? name.replace(new RegExp(`[-_.]${q}$`, "i"), "") : name;
};

export const gb = (bytes: number) => `${num(bytes / 1e9, bytes < 10e9 ? 1 : 0)} GB`;

/** Local models first, then each peer's, keeping their order. */
export function groupByNode(models: ModelInfo[]): [string | null, ModelInfo[]][] {
  const groups = new Map<string | null, ModelInfo[]>([[null, []]]);
  for (const m of models) groups.set(m.node ?? null, [...(groups.get(m.node ?? null) ?? []), m]);
  return [...groups].filter(([, ms]) => ms.length > 0);
}

/** The quantization to show: what the GGUF says, else what its file name says (older peers send none). */
export const quantLabel = (name: string, quant?: string) => quant || quantOf(name.replace(/(-\d{5}-of-\d{5})?\.gguf$/i, ""));

/** A count in thousands: `950`, `8.2k` (or `8,2k`), `33k`. */
export const thousands = (n: number) =>
  n < 1000 ? `${n}` : `${num(n / 1000, n < 10_000 ? 1 : 0)}k`;

/** A number as the UI's language writes it (`4.5` or `4,5`), with up to `digits` decimals. */
export const num = (n: number, digits = 0) =>
  n.toLocaleString(lang, { minimumFractionDigits: digits, maximumFractionDigits: digits });
