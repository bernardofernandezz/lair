import { useState } from "react";
import type { ToolStep } from "../parts";
import { domain, parseSources, type Source } from "./sources";
import { pick } from "../i18n";

const T = pick({
  en: { pending: "Waiting for you", running: "Running", ok: "Done", error: "Failed", denied: "Denied", awaiting: "Action waiting for approval", allow: "Allow", always: "Always in this chat", deny: "Deny", less: "Show less", all: "Show all" },
  pt: { pending: "Aguardando você", running: "Executando", ok: "Feito", error: "Falhou", denied: "Negado", awaiting: "Ação aguardando aprovação", allow: "Permitir", always: "Sempre nesta conversa", deny: "Negar", less: "Ver menos", all: "Ver tudo" },
});

const STATUS: Record<ToolStep["status"], string> = T;

/** The server names tools in English; Portuguese UIs show these instead. */
const LABEL = pick<Record<string, string>>({
  en: {},
  pt: {
    list_dir: "Listar pasta", read_file: "Ler arquivo", write_file: "Escrever arquivo", edit_file: "Editar arquivo",
    delete_file: "Apagar arquivo", run_command: "Rodar comando", web_search: "Buscar na web", fetch_url: "Abrir página",
    search_conversations: "Buscar conversas", read_conversation: "Ler conversa", list_skills: "Listar skills", save_skill: "Salvar skill",
  },
});

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
    <div className={`tool is-${step.status}`} role={pending ? "group" : undefined} aria-label={pending ? T.awaiting : undefined}>
      <details open={pending}>
        <summary>
          <span className="tool-dot" aria-hidden />
          <span className="tool-name">{LABEL[step.name] ?? step.label}</span>
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
            {T.allow}
          </button>
          <button className="ghost" disabled={deciding} onClick={() => decide(true, true)}>
            {T.always}
          </button>
          <button className="ghost" disabled={deciding} onClick={() => decide(false)}>
            {T.deny}
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

/** `fetch_url` output starts with `Title: …` when the page had one. */
function readPage(output?: string) {
  if (!output) return null;
  const m = output.match(/^(?:Title|Título): (.*)\n\n/);
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
          {all ? T.less : T.all}
        </button>
      )}
    </div>
  );
}
