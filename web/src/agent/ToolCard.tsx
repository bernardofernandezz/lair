import { useState } from "react";
import type { ToolStep } from "../parts";
import { domain, parseSources, type Source } from "./sources";

const STATUS: Record<ToolStep["status"], string> = {
  pending: "Aguardando você",
  running: "Executando",
  ok: "Feito",
  error: "Falhou",
  denied: "Negado",
};

type Props = {
  step: ToolStep;
  onDecide: (allow: boolean, always: boolean) => void;
};

export function ToolCard({ step, onDecide }: Props) {
  const [deciding, setDeciding] = useState(false);
  const pending = step.status === "pending";
  const detail = pending ? JSON.stringify(step.arguments, null, 2) : step.output;
  const results = !pending && step.name === "web_search" ? parseSources(step.output) : [];
  const page = !pending && step.name === "fetch_url" && step.status === "ok" ? readPage(step.output) : null;

  function decide(allow: boolean, always = false) {
    setDeciding(true);
    onDecide(allow, always);
  }

  return (
    <div className={`tool is-${step.status}`} role={pending ? "group" : undefined} aria-label={pending ? "Ação aguardando aprovação" : undefined}>
      <details open={pending}>
        <summary>
          <span className="tool-dot" aria-hidden />
          <span className="tool-name">{step.label}</span>
          <code className="tool-subject">{step.subject}</code>
          <span className="tool-status">{STATUS[step.status]}</span>
        </summary>
        {results.length > 0 ? (
          <SearchResults results={results} />
        ) : page ? (
          <PageText {...page} />
        ) : (
          detail && <pre className="tool-detail">{detail}</pre>
        )}
      </details>
      {pending && (
        <div className="tool-actions">
          <button className="solid" disabled={deciding} onClick={() => decide(true)}>
            Permitir
          </button>
          <button className="ghost" disabled={deciding} onClick={() => decide(true, true)}>
            Sempre nesta conversa
          </button>
          <button className="ghost" disabled={deciding} onClick={() => decide(false)}>
            Negar
          </button>
        </div>
      )}
    </div>
  );
}

function SearchResults({ results }: { results: Source[] }) {
  return (
    <ol className="search-results">
      {results.map((r) => (
        <li key={r.n}>
          <span className="search-n">{r.n}</span>
          <a href={r.url} target="_blank" rel="noreferrer noopener">
            <span className="search-domain">{domain(r.url)}</span>
            <span className="search-title">{r.title}</span>
          </a>
          {r.snippet && <p className="search-snippet">{r.snippet}</p>}
        </li>
      ))}
    </ol>
  );
}

/** `fetch_url` output starts with `Título: …` when the page had one. */
function readPage(output?: string) {
  if (!output) return null;
  const m = output.match(/^Título: (.*)\n\n/);
  return { title: m?.[1] ?? null, text: m ? output.slice(m[0].length) : output };
}

function PageText({ title, text }: { title: string | null; text: string }) {
  const [all, setAll] = useState(false);
  return (
    <div className="page-text">
      {title && <p className="page-title">{title}</p>}
      <pre className={`tool-detail${all ? "" : " is-clamped"}`}>{text}</pre>
      {text.length > 400 && (
        <button type="button" className="ghost page-more" onClick={() => setAll((a) => !a)}>
          {all ? "Ver menos" : "Ver tudo"}
        </button>
      )}
    </div>
  );
}
