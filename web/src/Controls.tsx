import { useEffect, useRef, useState, type CSSProperties, type FocusEvent, type KeyboardEvent } from "react";
import { modelName, type Effort, type ModelInfo } from "./api";

export const isEffort = (v: string): v is Effort => MODES.some((m) => m.value === v);

export const MODES: { value: Effort; label: string; hint: string }[] = [
  { value: "none", label: "Desligado", hint: "Responde direto, sem pensar" },
  { value: "low", label: "Leve", hint: "Pensa um pouco antes de responder" },
  { value: "high", label: "Profundo", hint: "Pensa o quanto precisar" },
];

// shortcut: read from the file name; read general.file_type from the GGUF if names outside the pattern show up.
const QUANT = /(?:^|[-_.])(I?Q\d(?:_[KS0-9])?(?:_[A-Z0-9]+)?|BF16|F16|F32)(?=$|[-_.])/i;

/** The quantization in a GGUF file name, like `Q4_K_M`, or null. */
export const quantOf = (id: string) => id.match(QUANT)?.[1].toUpperCase() ?? null;

/** The name without its quantization, which is shown apart as a tag. */
export const bareName = (name: string) => {
  const q = quantOf(name);
  return q ? name.replace(new RegExp(`[-_.]${q}$`, "i"), "") : name;
};

export const gb = (bytes: number) => `${(bytes / 1e9).toFixed(bytes < 10e9 ? 1 : 0)} GB`;

/** Local models first, then each peer's, keeping their order. */
export function groupByNode(models: ModelInfo[]): [string | null, ModelInfo[]][] {
  const groups = new Map<string | null, ModelInfo[]>([[null, []]]);
  for (const m of models) groups.set(m.node ?? null, [...(groups.get(m.node ?? null) ?? []), m]);
  return [...groups].filter(([, ms]) => ms.length > 0);
}

type Props = {
  models: ModelInfo[];
  value: string;
  onChange: (id: string) => void;
  effort: Effort;
  onEffort: (e: Effort) => void;
  disabled: boolean;
  /** Opens the place to download more models; the link is hidden without it. */
  onMore?: () => void;
};

/** One chip for the model and its reasoning: "Qwen3 8B · Profundo". */
export function ModelChip({ models, value, onChange, effort, onEffort, disabled, onMore }: Props) {
  const current = models.find((m) => m.id === value);
  const groups = groupByNode(models);
  const ordered = groups.flatMap(([, ms]) => ms);
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!open) return;
    setHi(Math.max(0, ordered.findIndex((m) => m.id === value)));
    list.current?.focus();
    const outside = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", outside);
    return () => window.removeEventListener("pointerdown", outside);
  }, [open]);

  function close() {
    setOpen(false);
    button.current?.focus();
  }

  function pick(i: number) {
    onChange(ordered[i].id);
    // A thinking model keeps the panel open so its reasoning can be set right away.
    if (!ordered[i].thinks) close();
  }

  function onKey(e: KeyboardEvent) {
    const last = ordered.length - 1;
    const moves: Record<string, () => void> = {
      ArrowDown: () => setHi((h) => Math.min(last, h + 1)),
      ArrowUp: () => setHi((h) => Math.max(0, h - 1)),
      Home: () => setHi(0),
      End: () => setHi(last),
      Enter: () => pick(hi),
      " ": () => pick(hi),
    };
    if (!moves[e.key]) return;
    e.preventDefault();
    moves[e.key]();
  }

  const mode = MODES.find((m) => m.value === effort);
  const name = bareName(current ? modelName(current) : value);
  let index = 0;

  return (
    <div
      className="picker"
      ref={root}
      onKeyDown={(e) => e.key === "Escape" && open && (e.preventDefault(), close())}
      onBlur={(e: FocusEvent) => !root.current?.contains(e.relatedTarget as Node) && setOpen(false)}
    >
      <button
        ref={button}
        type="button"
        className="chip model-chip"
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Modelo: ${name}${current?.thinks && mode ? `, raciocínio ${mode.label}` : ""}`}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="chip-label">{name}</span>
        {current?.thinks && mode && <span className="chip-mode">{mode.label}</span>}
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M7 10l5 5 5-5" />
        </svg>
      </button>
      {open && (
        <div className="picker-panel" role="dialog" aria-label="Modelo e raciocínio">
          <ul ref={list} role="listbox" tabIndex={0} aria-label="Modelo" aria-activedescendant={`model-${hi}`} onKeyDown={onKey}>
            {groups.map(([node, ms]) => [
              groups.length > 1 && (
                <li key={`h-${node}`} role="presentation" className="group-head">
                  {node ? `Em ${node}` : "Esta máquina"}
                </li>
              ),
              ...ms.map((m) => {
                const i = index++;
                const quant = quantOf(modelName(m));
                return (
                  <li
                    key={m.id}
                    id={`model-${i}`}
                    role="option"
                    aria-selected={m.id === value}
                    className={i === hi ? "is-hi" : undefined}
                    onPointerEnter={() => setHi(i)}
                    onClick={() => pick(i)}
                  >
                    <span className="option-name">{bareName(modelName(m))}</span>
                    <span className="option-meta">
                      {quant && <span className="tag">{quant}</span>}
                      {m.size ? <span className="tag">{gb(m.size)}</span> : null}
                      {m.thinks && <span className="tag">Pensa</span>}
                      {m.vision && <span className="tag">Visão</span>}
                      {m.loaded && <span className="tag is-on">Na memória</span>}
                    </span>
                  </li>
                );
              }),
            ])}
          </ul>
          {current?.thinks && (
            <div className="picker-section">
              <span className="picker-label" id="reasoning-label">
                Raciocínio
              </span>
              <ReasoningControl value={effort} onChange={onEffort} />
              <span className="picker-hint">{mode?.hint}</span>
            </div>
          )}
          {onMore && (
            <button
              type="button"
              className="picker-more"
              onClick={() => {
                setOpen(false);
                onMore();
              }}
            >
              Baixar mais modelos…
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function ReasoningControl({ value, onChange }: { value: Effort; onChange: (e: Effort) => void }) {
  const index = MODES.findIndex((m) => m.value === value);
  return (
    <div className="segmented" role="radiogroup" aria-labelledby="reasoning-label" style={{ "--i": index } as CSSProperties}>
      {MODES.map((m) => (
        <button key={m.value} type="button" role="radio" aria-checked={m.value === value} title={m.hint} onClick={() => onChange(m.value)}>
          {m.label}
        </button>
      ))}
    </div>
  );
}

/** How full the context was after the last reply; hidden until a reply reports it. */
export function ContextMeter({ used, size }: { used: number; size: number }) {
  if (!used || !size) return null;
  const share = Math.min(1, used / size);
  const k = (n: number) => (n < 1000 ? `${n}` : `${(n / 1000).toFixed(n < 10_000 ? 1 : 0).replace(".", ",")}k`);
  const label = `${k(used)} de ${k(size)} tokens de contexto usados`;
  return (
    <span className={`context-meter${share > 0.8 ? " is-full" : ""}`} role="img" aria-label={label} title={label}>
      <svg viewBox="0 0 20 20" aria-hidden>
        <circle cx="10" cy="10" r="8" />
        <circle cx="10" cy="10" r="8" pathLength="100" strokeDasharray={`${share * 100} 100`} />
      </svg>
    </span>
  );
}
