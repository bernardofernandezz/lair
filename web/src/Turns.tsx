import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ComponentProps } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToolCard } from "./agent/ToolCard";
import { citedIn, domain, linkCitations, sourcesOf, type Source } from "./agent/sources";
import { MAX_CANVASES, SceneCard } from "./scene/SceneCard";
import { splitMessage, type Segment } from "./scene/split";
import { convPath } from "./api";
import { hasText, type AssistantTurn, type Attached, type Turn } from "./parts";

type Props = {
  turns: Turn[];
  /** The last turn is still streaming. */
  live: boolean;
  /** Name of the model being loaded for this reply, if it is not in memory yet. */
  loading: string | false;
  onDecide: (callId: string, allow: boolean, always: boolean) => void;
  /** Fetches a stored attachment as a blob URL (the token can't ride on an `<img>`). */
  fileUrl: (id: string) => Promise<string>;
  onOpen: (id: string) => void;
};

/** An attached image on a user turn: the local preview right after sending, the stored file later. */
function Thumb({ a, fileUrl }: { a: Attached; fileUrl: Props["fileUrl"] }) {
  const [src, setSrc] = useState(a.src ?? null);
  useEffect(() => {
    if (a.src || !a.id) return;
    let url: string | null = null;
    let live = true;
    fileUrl(a.id).then((u) => (live ? setSrc((url = u)) : URL.revokeObjectURL(u)), () => {});
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [a.src, a.id, fileUrl]);
  return (
    <span className="thumb" title={a.name}>
      {src ? <img src={src} alt={a.name} loading="lazy" /> : <span className="thumb-missing">{a.name}</span>}
    </span>
  );
}

export function Turns({ turns, live, loading, onDecide, fileUrl, onOpen }: Props) {
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
              {t.attachments?.some((a) => a.image) && (
                <span className="thumbs">
                  {t.attachments.filter((a) => a.image).map((a, j) => (
                    <Thumb key={`${j}-${a.name}`} a={a} fileUrl={fileUrl} />
                  ))}
                </span>
              )}
              {(!!t.refs?.length || t.attachments?.some((a) => !a.image)) && (
                <span className="ref-chips">
                  {t.refs?.map((r) => (
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
                  {t.attachments?.filter((a) => !a.image).map((a, j) => (
                    <span key={`${j}-${a.name}`} className={`ref-chip is-file`} title="Documento anexado à mensagem">
                      <svg viewBox="0 0 24 24" aria-hidden>
                        <path d="M7 5h7l5 5v9a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1zM13 5v6h6" />
                      </svg>
                      <span>{a.name}</span>
                    </span>
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
  const sources = useMemo(() => sourcesOf(turn.parts), [turn.parts]);
  const text = turn.parts.map((p) => (p.kind === "text" ? p.text : "")).join("\n");
  const cited = citedIn(text, sources);
  // Small models often skip citing: then every result searched is listed.
  const listed = (cited.length ? cited : [...sources.keys()]).map((n) => sources.get(n)!);
  const components = useMemo(
    () => ({
      a: ({ href, children, node: _, ...rest }: ComponentProps<"a"> & { node?: unknown }) => {
        const src = href?.startsWith("#src-") ? sources.get(Number(href.slice(5))) : undefined;
        return src ? <Cite source={src} /> : <a href={href} {...rest}>{children}</a>;
      },
    }),
    [sources],
  );
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
                  <Markdown key={k} remarkPlugins={[remarkGfm]} components={components}>
                    {linkCitations(segment.text, sources)}
                  </Markdown>
                ) : (
                  <SceneCard key={k} source={segment.source} closed={segment.closed} render={scene++ < MAX_CANVASES} />
                ),
              )}
            </div>
          )
        ),
      )}
      {!live && listed.length > 0 && (
        <p className="sources" aria-label="Fontes">
          <span className="sources-label">Fontes</span>
          {listed.map((s) => (
            <Cite key={s.n} source={s} />
          ))}
        </p>
      )}
    </li>
  );
}

/** A source as a small badge with its domain, linking to the page. */
function Cite({ source }: { source: Source }) {
  return (
    <a className="cite" href={source.url} target="_blank" rel="noreferrer noopener" title={`${source.n}. ${source.title}`}>
      {domain(source.url)}
    </a>
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
