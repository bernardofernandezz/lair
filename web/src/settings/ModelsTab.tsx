import { useCallback, useEffect, useRef, useState } from "react";
import type { Api, Download, Fit, Hardware, HubFile, HubRepo, Suggestion } from "../api";
import { gb, quantOf } from "../Controls";

const FIT: Record<Fit, { label: string; hint: string }> = {
  fits: { label: "Cabe", hint: "Roda inteiro na GPU (ou na RAM, sem GPU)." },
  tight: { label: "Apertado", hint: "Roda com parte das camadas na CPU: funciona, mais devagar." },
  no: { label: "Não cabe", hint: "Maior que a memória desta máquina." },
};

const active = (d: Download) => !d.finished && !d.error;

/** Find models on Hugging Face, see which fit this machine, and download them. */
export function ModelsTab({ api, onInstalled }: { api: Api; onInstalled: () => void }) {
  const [hw, setHw] = useState<Hardware | null>(null);
  const [suggested, setSuggested] = useState<Suggestion[] | null>(null);
  const [downloads, setDownloads] = useState<Download[]>([]);
  const [query, setQuery] = useState("");
  const [repos, setRepos] = useState<HubRepo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped when a download finishes, so file lists show what is now installed.
  const [installed, setInstalled] = useState(0);
  const finished = useRef(new Set<string>());

  const fail = (e: unknown) => setError((e as Error).message);

  useEffect(() => {
    api.hardware().then(setHw, fail);
    api.downloads().then(setDownloads, fail);
  }, [api]);
  useEffect(() => {
    api.suggested().then(setSuggested, fail);
  }, [api, installed]);

  // Polls only while something is downloading.
  const polling = downloads.some(active);
  useEffect(() => {
    if (!polling) return;
    const t = setInterval(() => api.downloads().then(setDownloads, () => {}), 1000);
    return () => clearInterval(t);
  }, [api, polling]);
  useEffect(() => {
    const done = downloads.filter((d) => d.finished && !finished.current.has(d.file));
    if (!done.length) return;
    done.forEach((d) => finished.current.add(d.file));
    setInstalled((n) => n + 1);
    onInstalled();
  }, [downloads, onInstalled]);

  useEffect(() => {
    const q = query.trim();
    if (!q) return setRepos(null);
    const t = setTimeout(() => api.searchHub(q).then(setRepos, fail), 400);
    return () => clearTimeout(t);
  }, [api, query]);

  const start = useCallback(
    async (repo: string, file: string) => {
      setError(null);
      try {
        await api.download(repo, file);
        setDownloads(await api.downloads());
      } catch (e) {
        fail(e);
      }
    },
    [api],
  );

  const cancel = async (file: string) => {
    try {
      await api.cancelDownload(file);
      setDownloads(await api.downloads());
    } catch (e) {
      fail(e);
    }
  };

  const running = (file: string) => downloads.find((d) => d.file === file && active(d));

  return (
    <div className="models-tab">
      <p className="hw-line">
        {hw ? (
          <>
            {hw.gpus.length ? hw.gpus.join(", ") : "Sem GPU detectada"}
            {hw.vram > 0 && <span className="tag">{gb(hw.vram)} de VRAM</span>}
            {hw.ram > 0 && <span className="tag">{gb(hw.ram)} de RAM</span>}
          </>
        ) : (
          "Lendo o hardware…"
        )}
      </p>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {downloads.length > 0 && (
        <section aria-label="Downloads">
          <h3 className="models-head">Downloads</h3>
          <ul className="skill-list">
            {downloads.map((d) => (
              <li key={d.file} className="download">
                <div className="skill-main">
                  <span className="skill-name">{d.file}</span>
                  <span className="skill-desc">
                    {d.error ? d.error : d.finished ? "Pronto: já aparece na lista de modelos." : `${gb(d.done)} de ${gb(d.total)}`}
                  </span>
                  {active(d) && (
                    <progress max={d.total || 1} value={d.done} aria-label={`Baixando ${d.file}`} />
                  )}
                </div>
                <div className="skill-actions">
                  <button className="ghost" onClick={() => cancel(d.file)}>
                    {active(d) ? "Cancelar" : "Dispensar"}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-label="Recomendados">
        <h3 className="models-head">Recomendados para esta máquina</h3>
        {!suggested && <p className="hint">Consultando o Hugging Face…</p>}
        {suggested?.length === 0 && <p className="hint">Não foi possível falar com o Hugging Face agora.</p>}
        <ul className="skill-list">
          {suggested?.map((s) => (
            <li key={s.repo} className={s.best ? "is-best" : undefined}>
              <div className="skill-main">
                <span className="skill-name">
                  {s.repo.split("/")[1].replace(/-GGUF$/i, "")}
                  {s.best && <span className="badge is-live">Melhor escolha</span>}
                </span>
                <span className="skill-desc">{s.about}</span>
                {s.pick && <FileTags file={s.pick} vision={s.vision} />}
              </div>
              <div className="skill-actions">
                {s.pick ? (
                  <DownloadButton file={s.pick} running={running(s.pick.name)} onStart={() => start(s.repo, s.pick!.name)} />
                ) : (
                  <span className="hint">Não cabe</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section aria-label="Buscar no Hugging Face">
        <h3 className="models-head">Buscar no Hugging Face</h3>
        <div className="field">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ex.: qwen3, llama, mistral, gemma"
            aria-label="Buscar modelos GGUF no Hugging Face"
          />
        </div>
        <p className="hint">Só repositórios com arquivos .gguf. Modelos com acesso restrito ou divididos em partes não aparecem.</p>
        {repos?.length === 0 && <p className="hint">Nada encontrado.</p>}
        {repos && repos.length > 0 && (
          <ul className="repo-list">
            {repos.map((r) => (
              <RepoRow key={r.id} api={api} repo={r} installed={installed} running={running} onStart={start} />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

function FileTags({ file, vision }: { file: HubFile; vision?: boolean }) {
  const quant = quantOf(file.name.replace(/\.gguf$/i, ""));
  return (
    <span className="option-meta">
      {quant && <span className="tag">{quant}</span>}
      <span className="tag">{gb(file.size)}</span>
      <span className={`tag fit-${file.fit}`} title={FIT[file.fit].hint}>
        {FIT[file.fit].label}
      </span>
      {vision && <span className="tag">Visão</span>}
    </span>
  );
}

function DownloadButton({ file, running, onStart }: { file: HubFile; running?: Download; onStart: () => void }) {
  if (file.installed) return <span className="badge">Instalado</span>;
  if (running) return <span className="hint">{Math.round((running.done / (running.total || 1)) * 100)}%</span>;
  return (
    <button className={file.fit === "no" ? "ghost" : "solid"} onClick={onStart} title={FIT[file.fit].hint}>
      Baixar
    </button>
  );
}

function RepoRow({
  api,
  repo,
  installed,
  running,
  onStart,
}: {
  api: Api;
  repo: HubRepo;
  installed: number;
  running: (file: string) => Download | undefined;
  onStart: (repo: string, file: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [files, setFiles] = useState<{ files: HubFile[]; pick: string | null } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    api.hubFiles(repo.id).then(setFiles, (e) => setError((e as Error).message));
  }, [api, repo.id, open, installed]);

  const models = files?.files.filter((f) => !f.mmproj) ?? [];
  const vision = files?.files.some((f) => f.mmproj);

  return (
    <li>
      <button className="repo-head" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
        <span className="skill-name">{repo.id}</span>
        <span className="skill-desc">{repo.downloads.toLocaleString("pt-BR")} downloads</span>
      </button>
      {open && (
        <div className="repo-files">
          {error && <p className="error">{error}</p>}
          {!files && !error && <p className="hint">Listando arquivos…</p>}
          {files && models.length === 0 && <p className="hint">Nenhum .gguf de arquivo único neste repositório.</p>}
          <ul className="skill-list">
            {models.map((f) => (
              <li key={f.name}>
                <div className="skill-main">
                  <span className="skill-name">
                    {f.name}
                    {f.name === files?.pick && <span className="badge is-live">Recomendado</span>}
                  </span>
                  <FileTags file={f} vision={vision} />
                </div>
                <div className="skill-actions">
                  <DownloadButton file={f} running={running(f.name)} onStart={() => onStart(repo.id, f.name)} />
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </li>
  );
}
