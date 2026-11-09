import { useEffect, useRef, useState, type CSSProperties, type KeyboardEvent } from "react";
import type { Effort, ModelInfo } from "./api";

type PickerProps = { models: ModelInfo[]; value: string; onChange: (id: string) => void; disabled: boolean };

export function ModelPicker({ models, value, onChange, disabled }: PickerProps) {
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);

  useEffect(() => {
    if (!open) return;
    setHi(Math.max(0, models.findIndex((m) => m.id === value)));
    list.current?.focus();
    const outside = (e: PointerEvent) => {
      if (!root.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("pointerdown", outside);
    return () => window.removeEventListener("pointerdown", outside);
  }, [open, models, value]);

  function pick(i: number) {
    onChange(models[i].id);
    setOpen(false);
    button.current?.focus();
  }

  function onKey(e: KeyboardEvent) {
    const last = models.length - 1;
    const moves: Record<string, () => void> = {
      ArrowDown: () => setHi((h) => Math.min(last, h + 1)),
      ArrowUp: () => setHi((h) => Math.max(0, h - 1)),
      Home: () => setHi(0),
      End: () => setHi(last),
      Enter: () => pick(hi),
      " ": () => pick(hi),
      Escape: () => {
        setOpen(false);
        button.current?.focus();
      },
      Tab: () => setOpen(false),
    };
    if (!moves[e.key]) return;
    if (e.key !== "Tab") e.preventDefault();
    moves[e.key]();
  }

  return (
    <div className="picker" ref={root}>
      <button
        ref={button}
        type="button"
        className="chip"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Modelo: ${value}`}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="chip-label">{value}</span>
        <svg viewBox="0 0 24 24" aria-hidden>
          <path d="M7 10l5 5 5-5" />
        </svg>
      </button>
      {open && (
        <ul
          ref={list}
          role="listbox"
          tabIndex={-1}
          aria-label="Modelo"
          aria-activedescendant={`model-${hi}`}
          onKeyDown={onKey}
        >
          {models.map((m, i) => (
            <li
              key={m.id}
              id={`model-${i}`}
              role="option"
              aria-selected={m.id === value}
              className={i === hi ? "is-hi" : undefined}
              onPointerEnter={() => setHi(i)}
              onClick={() => pick(i)}
            >
              <span className="option-name">{m.id}</span>
              <span className="option-meta">
                {m.thinks ? "Raciocina" : "Responde direto"}
                {m.loaded && ", já na memória"}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export const isEffort = (v: string): v is Effort => MODES.some((m) => m.value === v);

const MODES: { value: Effort; label: string; hint: string }[] = [
  { value: "none", label: "Desligado", hint: "Responde direto, sem pensar" },
  { value: "low", label: "Leve", hint: "Pensa um pouco antes de responder" },
  { value: "high", label: "Profundo", hint: "Pensa o quanto precisar" },
];

type ReasoningProps = { value: Effort; onChange: (e: Effort) => void; available: boolean };

export function ReasoningControl({ value, onChange, available }: ReasoningProps) {
  const index = MODES.findIndex((m) => m.value === value);
  return (
    <div
      className="segmented"
      role="radiogroup"
      aria-label="Raciocínio"
      title={available ? undefined : "Este modelo responde sem raciocinar"}
      style={{ "--i": index } as CSSProperties}
      aria-disabled={!available}
    >
      {MODES.map((m) => (
        <button
          key={m.value}
          type="button"
          role="radio"
          aria-checked={m.value === value}
          title={available ? m.hint : undefined}
          disabled={!available}
          onClick={() => onChange(m.value)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
