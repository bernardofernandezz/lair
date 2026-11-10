import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToolCard } from "./agent/ToolCard";
import { MAX_CANVASES, SceneCard } from "./scene/SceneCard";
import { splitMessage, type Segment } from "./scene/split";
import { convPath } from "./api";
import { hasText, type AssistantTurn, type Turn } from "./parts";

type Props = {
  turns: Turn[];
  /** The last turn is still streaming. */
  live: boolean;
  /** Name of the model being loaded for this reply, if it is not in memory yet. */
  loading: string | false;
  onDecide: (callId: string, allow: boolean, always: boolean) => void;
  onOpen: (id: string) => void;
};

export function Turns({ turns, live, loading, onDecide, onOpen }: Props) {
  const segments = useMemo(
    () => turns.map((t) => (t.role === "assistant" ? t.parts.map((p) => (p.kind === "text" ? splitMessage(p.text) : [])) : [])),
    [turns],
  );
  let scenes = 0;
  return (
    <ol className="turns">
      {turns.map((t, i) => {
        if (t.role === "user") {
          return (
            <li key={i} className="turn user">
              {!!t.refs?.length && (
                <span className="ref-chips">
                  {t.refs.map((r) => (
                    <a
                      key={r.id}
                      className="ref-chip"
                      href={convPath(r.id)}
                      onClick={(e) => {
                        if (e.metaKey || e.ctrlKey || e.shiftKey) return;
                        e.preventDefault();
                        onOpen(r.id);
                      }}
                    >
                      <span>{r.title}</span>
                    </a>
                  ))}
                </span>
              )}
              {t.content}
            </li>
          );
        }
        const first = scenes;
        scenes += segments[i].flat().filter((s) => s.kind === "scene").length;
        const isLive = live && i === turns.length - 1;
        return (
          <Reply
            key={i}
            turn={t}
            segments={segments[i]}
            sceneStart={first}
            live={isLive}
            loading={loading}
            onDecide={onDecide}
          />
        );
      })}
    </ol>
  );
}

function Reply({
  turn,
  segments,
  sceneStart,
  live,
  loading,
  onDecide,
}: {
  turn: AssistantTurn;
  segments: Segment[][];
  sceneStart: number;
  live: boolean;
  loading: string | false;
  onDecide: Props["onDecide"];
}) {
  const thinking = live && !hasText(turn) && !turn.parts.some((p) => p.kind === "tool");
  let scene = sceneStart;
  return (
    <li className="turn assistant" aria-busy={live}>
      {live && !turn.reasoning && turn.parts.length === 0 && (
        <p className="status">{loading ? `Carregando ${loading}` : "Preparando a resposta"}</p>
      )}
      {turn.reasoning && <Reasoning text={turn.reasoning} thinking={thinking} ms={turn.thinkMs} />}
      {turn.skill && (
        <p className="skills" aria-label="Skill ativada">
          <span className="skill">
            <span className="skill-dot" aria-hidden />
            {turn.skill.name}
          </span>
        </p>
      )}
      {turn.parts.map((part, j) =>
        part.kind === "tool" ? (
          <ToolCard key={part.step.id} step={part.step} onDecide={(allow, always) => onDecide(part.step.id, allow, always)} />
        ) : (
          part.text.trim() && (
            <div key={j} className="answer" aria-live={live ? "polite" : undefined}>
              {segments[j].map((segment, k) =>
                segment.kind === "md" ? (
                  <Markdown key={k} remarkPlugins={[remarkGfm]}>
                    {segment.text}
                  </Markdown>
                ) : (
                  <SceneCard key={k} source={segment.source} closed={segment.closed} render={scene++ < MAX_CANVASES} />
                ),
              )}
            </div>
          )
        ),
      )}
    </li>
  );
}

/** The model's reasoning, kept visually apart from its answer: open while it thinks, folded once it answers. */
function Reasoning({ text, thinking, ms }: { text: string; thinking: boolean; ms?: number }) {
  const [open, setOpen] = useState(thinking);
  const body = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!thinking) setOpen(false);
  }, [thinking]);
  useLayoutEffect(() => {
    if (thinking && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [text, thinking]);

  const label = thinking ? "Pensando" : ms ? `Pensou por ${Math.max(1, Math.round(ms / 1000))} s` : "Raciocínio";
  return (
    <details className={`reasoning${thinking ? " is-live" : ""}`} open={open} onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary>
        <span className={thinking ? "thinking" : undefined}>{label}</span>
      </summary>
      <div className="reasoning-body" ref={body}>
        {text.trim()}
      </div>
    </details>
  );
}
