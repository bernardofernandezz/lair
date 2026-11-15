import { useEffect, useRef, useState, type CSSProperties, type FocusEvent, type KeyboardEvent } from "react";
import { AUTO, modelName, type Effort, type ModelInfo } from "./api";
import { pick } from "./i18n";
import { ModelTags } from "./ModelTags";
import { bareName, displayName, groupByNode, MODES, thousands } from "./models";

const T = pick({
  en: {
    model: "Model",
    reasoningOf: (m: string) => `, reasoning ${m}`,
    modelAndReasoning: "Model and reasoning",
    on: (n: string) => `On ${n}`,
    here: "This machine",
    reasoning: "Reasoning",
    more: "Get more models…",
    used: (u: string, s: string) => `${u} of ${s} context tokens used`,
    autoHint: "lair picks per message: the biggest model that is fast right now, here or on your other machines",
  },
  pt: {
    model: "Modelo",
    reasoningOf: (m: string) => `, raciocínio ${m}`,
    modelAndReasoning: "Modelo e raciocínio",
    on: (n: string) => `Em ${n}`,
    here: "Esta máquina",
    reasoning: "Raciocínio",
    more: "Baixar mais modelos…",
    used: (u: string, s: string) => `${u} de ${s} tokens de contexto usados`,
    autoHint: "o lair escolhe a cada mensagem: o maior modelo que está rápido agora, aqui ou nas suas outras máquinas",
  },
});

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
  const name = current ? displayName(current) : bareName(value);
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
        aria-label={`${T.model}: ${name}${current?.thinks && mode ? T.reasoningOf(mode.label) : ""}`}
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
        <div className="picker-panel" role="dialog" aria-label={T.modelAndReasoning}>
          <ul ref={list} role="listbox" tabIndex={0} aria-label={T.model} aria-activedescendant={`model-${hi}`} onKeyDown={onKey}>
            {groups.map(([node, ms]) => [
              groups.length > 1 && (
                <li key={`h-${node}`} role="presentation" className="group-head">
                  {node ? T.on(node) : T.here}
                </li>
              ),
              ...ms.map((m) => {
                const i = index++;
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
                    <span className="option-name" title={m.id === AUTO ? T.autoHint : undefined}>
                      {displayName(m)}
                    </span>
                    <ModelTags {...m} name={modelName(m)} />
                  </li>
                );
              }),
            ])}
          </ul>
          {current?.thinks && (
            <div className="picker-section">
              <span className="picker-label" id="reasoning-label">
                {T.reasoning}
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
              {T.more}
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
  const label = T.used(thousands(used), thousands(size));
  return (
    <span className={`context-meter${share > 0.8 ? " is-full" : ""}`} role="img" aria-label={label} title={label}>
      <svg viewBox="0 0 20 20" aria-hidden>
        <circle cx="10" cy="10" r="8" />
        <circle cx="10" cy="10" r="8" pathLength="100" strokeDasharray={`${share * 100} 100`} />
      </svg>
    </span>
  );
}
