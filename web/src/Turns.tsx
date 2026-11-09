import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Ref, SkillInfo } from "./api";
import { MAX_CANVASES, SceneCard } from "./scene/SceneCard";
import { splitMessage } from "./scene/split";

export type Turn = {
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  thinkMs?: number;
  skill?: SkillInfo | null;
  refs?: Ref[];
};

type Props = {
  turns: Turn[];
  /** The last turn is still streaming. */
  live: boolean;
  /** Name of the model being loaded for this reply, if it is not in memory yet. */
  loading: string | false;
};

export function Turns({ turns, live, loading }: Props) {
  const segments = useMemo(() => turns.map((t) => (t.role === "assistant" ? splitMessage(t.content) : [])), [turns]);
  let scenes = 0;
  return (
    <ol className="turns">
      {turns.map((t, i) => {
        if (t.role === "user") return <UserTurn key={i} turn={t} />;
        const first = scenes;
        scenes += segments[i].filter((s) => s.kind === "scene").length;
        const isLive = live && i === turns.length - 1;
        const thinking = isLive && !t.content;
        let scene = first;
        return (
          <li key={i} className="turn assistant" aria-busy={isLive}>
            {isLive && !t.reasoning && !t.content && (
              <p className="status">{loading ? `Carregando ${loading}` : "Preparando a resposta"}</p>
            )}
            {t.reasoning && <Reasoning text={t.reasoning} thinking={thinking} ms={t.thinkMs} />}
            {t.skill && (
              <p className="skills" aria-label="Skill ativada">
                <span className="skill">
                  <span className="skill-dot" aria-hidden />
                  {t.skill.name}
                </span>
              </p>
            )}
            {t.content && (
              <div className="answer" aria-live={isLive ? "polite" : undefined}>
                {segments[i].map((segment, j) =>
                  segment.kind === "md" ? (
                    <Markdown key={j} remarkPlugins={[remarkGfm]}>
                      {segment.text}
                    </Markdown>
                  ) : (
                    <SceneCard key={j} source={segment.source} closed={segment.closed} render={scene++ < MAX_CANVASES} />
                  ),
                )}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

function UserTurn({ turn }: { turn: Turn }) {
  return (
    <li className="turn user">
      {!!turn.refs?.length && (
        <span className="ref-chips">
          {turn.refs.map((r) => (
            <a key={r.id} className="ref-chip" href={`/c/${r.id}`}>
              <span>{r.title}</span>
            </a>
          ))}
        </span>
      )}
      {turn.content}
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
