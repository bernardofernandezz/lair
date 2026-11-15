import type { ReactNode } from "react";
import { pick } from "./i18n";
import { gb, quantLabel, speedHint, speedLabel } from "./models";

const T = pick({
  en: { thinks: "Thinks", vision: "Vision", loaded: "In memory", embeds: "Embeddings" },
  pt: { thinks: "Pensa", vision: "Visão", loaded: "Na memória", embeds: "Embeddings" },
});

type Props = {
  /** The model's or file's name, for a quantization the GGUF doesn't state. */
  name: string;
  quant?: string;
  size?: number;
  tps?: number;
  measured?: boolean;
  thinks?: boolean;
  vision?: boolean;
  loaded?: boolean;
  embedding?: boolean;
  /** Tags about where it can come from, after its size. */
  children?: ReactNode;
};

/** A model's tags: quantization, size, speed and what it can do. */
export function ModelTags({ name, quant, size, tps, measured, thinks, vision, loaded, embedding, children }: Props) {
  const q = quantLabel(name, quant);
  const speed = speedLabel(tps, measured);
  return (
    <span className="option-meta">
      {q && <span className="tag">{q}</span>}
      {size ? <span className="tag">{gb(size)}</span> : null}
      {children}
      {speed && (
        <span className="tag" title={speedHint(measured)}>
          {speed}
        </span>
      )}
      {embedding && <span className="tag">{T.embeds}</span>}
      {thinks && <span className="tag">{T.thinks}</span>}
      {vision && <span className="tag">{T.vision}</span>}
      {loaded && <span className="tag is-on">{T.loaded}</span>}
    </span>
  );
}
