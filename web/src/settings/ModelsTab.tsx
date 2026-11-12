import { useCallback, useEffect, useRef, useState } from "react";
import type { Api, Download, Fit, Hardware, HubFile, HubRepo, LoadedModel, ModelInfo, Suggestion, Why } from "../api";
import { bareName, gb, quantOf, speedHint, speedLabel } from "../Controls";
import { pick } from "../i18n";

const T = pick({
  en: {
    fit: {
      fits: { label: "Fits", hint: "Runs fully on the GPU (or in RAM, without a GPU)." },
      tight: { label: "Tight", hint: "Runs with some layers on the CPU: works, but slower." },
      no: { label: "Doesn't fit", hint: "Bigger than this machine's memory." },
    },
    noGpu: "No GPU found", vram: " VRAM", ram: " RAM", reading: "Reading the hardware…",
    free: (a: string, b: string) => `${a} free of ${b}`, cores: (n: number) => `${n} threads`,
    runsOn: { cuda: "Runs on CUDA", vulkan: "Runs on Vulkan", metal: "Runs on Metal", cpu: "Runs on the CPU" },
    wsl: "Inside WSL, where lair runs on the CPU: the GPU is only reachable through a slow emulation.",
    ready: "Done: it's in the model list now.", of: (a: string, b: string) => `${a} of ${b}`, downloading: (f: string) => `Downloading ${f}`,
    verifying: "Checking the download against its published hash…", resume: "Continue",
    resumeHint: "What was already downloaded is kept; this continues from there.",
    cancel: "Cancel", dismiss: "Dismiss", recommended: "Recommended", forThis: "Best for this machine, from Hugging Face",
    forThisHint: "What is trending on Hugging Face and what lair vouches for, judged against this machine. The best pick is the biggest model from a trusted publisher that runs fully here, fast, at 4 bits or better.",
    asking: "Asking Hugging Face…", hfDown: "Couldn't reach Hugging Face right now.", best: "Best pick", noFit: "Doesn't fit",
    why: {
      fits: "runs fully on this machine", tight: "runs partly on the CPU", fast: "fast here", slow: "slow here",
      trusted: "trusted publisher", trending: "trending", gated: "needs a Hugging Face token", low_quant: "heavily compressed",
    },
    search: "Search Hugging Face", searchPlaceholder: "E.g. qwen3, llama, mistral, gemma", searchLabel: "Search GGUF models on Hugging Face",
    searchHint: "Only repos with .gguf files. Gated and private repos need a Hugging Face token (General tab).", nothing: "Nothing found.",
    vision: "Vision", thinks: "Thinks", installed: "Installed", download: "Download", listing: "Listing files…", noFiles: "No .gguf model in this repo.",
    parts: (n: number) => `${n} parts`,
    onDisk: "On this machine", none: "No models yet. Pick one below.", inMemory: "In memory", embeds: "Embeddings",
    loadedAs: (g: number, l: number, c: number) => `${g} of ${l} layers on the GPU, context ${c.toLocaleString()}`,
    unload: "Unload", unloadHint: "Frees its memory; it loads again when a chat uses it.",
    remove: "Delete", confirmRemove: (m: string) => `Delete ${m} from this machine? Its file is removed from disk.`,
    freed: (s: string) => `Deleted; ${s} freed.`,
  },
  pt: {
    fit: {
      fits: { label: "Cabe", hint: "Roda inteiro na GPU (ou na RAM, sem GPU)." },
      tight: { label: "Apertado", hint: "Roda com parte das camadas na CPU: funciona, mais devagar." },
      no: { label: "Não cabe", hint: "Maior que a memória desta máquina." },
    },
    noGpu: "Sem GPU detectada", vram: " de VRAM", ram: " de RAM", reading: "Lendo o hardware…",
    free: (a: string, b: string) => `${a} livres de ${b}`, cores: (n: number) => `${n} threads`,
    runsOn: { cuda: "Roda em CUDA", vulkan: "Roda em Vulkan", metal: "Roda em Metal", cpu: "Roda na CPU" },
    wsl: "Dentro do WSL, onde o lair roda na CPU: a GPU só é alcançada por uma emulação lenta.",
    ready: "Pronto: já aparece na lista de modelos.", of: (a: string, b: string) => `${a} de ${b}`, downloading: (f: string) => `Baixando ${f}`,
    verifying: "Conferindo o download com o hash publicado…", resume: "Continuar",
    resumeHint: "O que já foi baixado fica guardado; o download continua de onde parou.",
    cancel: "Cancelar", dismiss: "Dispensar", recommended: "Recomendado", forThis: "Melhores para esta máquina, do Hugging Face",
    forThisHint: "O que está em alta no Hugging Face e o que o lair recomenda, avaliados para esta máquina. A melhor escolha é o maior modelo de um publicador confiável que roda inteiro aqui, rápido, com 4 bits ou mais.",
    asking: "Consultando o Hugging Face…", hfDown: "Não foi possível falar com o Hugging Face agora.", best: "Melhor escolha", noFit: "Não cabe",
    why: {
      fits: "roda inteiro nesta máquina", tight: "roda em parte na CPU", fast: "rápido aqui", slow: "lento aqui",
      trusted: "publicador confiável", trending: "em alta", gated: "precisa de token do Hugging Face", low_quant: "muito comprimido",
    },
    search: "Buscar no Hugging Face", searchPlaceholder: "Ex.: qwen3, llama, mistral, gemma", searchLabel: "Buscar modelos GGUF no Hugging Face",
    searchHint: "Só repositórios com arquivos .gguf. Repositórios restritos ou privados precisam de um token do Hugging Face (aba Geral).", nothing: "Nada encontrado.",
    vision: "Visão", thinks: "Pensa", installed: "Instalado", download: "Baixar", listing: "Listando arquivos…", noFiles: "Nenhum modelo .gguf neste repositório.",
    parts: (n: number) => `${n} partes`,
    onDisk: "Nesta máquina", none: "Nenhum modelo ainda. Escolha um abaixo.", inMemory: "Na memória", embeds: "Embeddings",
    loadedAs: (g: number, l: number, c: number) => `${g} de ${l} camadas na GPU, contexto ${c.toLocaleString()}`,
    unload: "Descarregar", unloadHint: "Libera a memória; ele carrega de novo quando uma conversa usar.",
    remove: "Apagar", confirmRemove: (m: string) => `Apagar ${m} desta máquina? O arquivo é removido do disco.`,
    freed: (s: string) => `Apagado; ${s} liberados.`,
  },
});

const FIT: Record<Fit, { label: string; hint: string }> = T.fit;

const active = (d: Download) => !d.finished && !d.error;
const keyOf = (d: { repo: string; file: string }) => `${d.repo}/${d.file}`;

/** A suggestion's reasons as one line: "trusted publisher · runs fully on this machine · fast here". */
export const reasonLine = (why: Why[], words: Record<Why, string> = T.why) => why.map((w) => words[w]).filter(Boolean).join(" · ");

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
  const [local, setLocal] = useState<{ models: ModelInfo[]; loaded: LoadedModel[] } | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const fail = (e: unknown) => setError((e as Error).message);

  useEffect(() => {
    api.hardware().then(setHw, fail);
    api.downloads().then(setDownloads, fail);
  }, [api]);
  useEffect(() => {
    api.suggested().then(setSuggested, (e) => (setSuggested([]), fail(e)));
    api.localModels().then(setLocal, fail);
    api.hardware().then(setHw, () => {});
  }, [api, installed]);

  // Polls only while something is downloading.
  const polling = downloads.some(active);
  useEffect(() => {
    if (!polling) return;
    const t = setInterval(() => api.downloads().then(setDownloads, () => {}), 1000);
    return () => clearInterval(t);
  }, [api, polling]);
  useEffect(() => {
    const done = downloads.filter((d) => d.finished && !finished.current.has(keyOf(d)));
    if (!done.length) return;
    done.forEach((d) => finished.current.add(keyOf(d)));
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

  const cancel = async (d: Download) => {
    try {
      await api.cancelDownload(d.repo, d.file);
      setDownloads(await api.downloads());
    } catch (e) {
      fail(e);
    }
  };

  /** Runs a change to the installed models, then shows them (and the memory they use) again. */
  const change = async (what: () => Promise<string | void>) => {
    setError(null);
    setNotice(null);
    try {
      setNotice((await what()) ?? null);
      setInstalled((n) => n + 1);
      onInstalled();
    } catch (e) {
      fail(e);
    }
  };
  const remove = (m: ModelInfo) => {
    if (!confirm(T.confirmRemove(m.id))) return;
    change(async () => T.freed(gb((await api.removeModel(m.id)).freed)));
  };

  const running = (repo: string, file: string) => downloads.find((d) => d.repo === repo && d.file === file && active(d));

  return (
    <div className="models-tab">
      <HardwarePanel hw={hw} />

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {notice && (
        <p className="hint" role="status">
          {notice}
        </p>
      )}

      <section aria-label={T.onDisk}>
        <h3 className="models-head">{T.onDisk}</h3>
        {local?.models.length === 0 && <p className="hint">{T.none}</p>}
        <ul className="skill-list">
          {local?.models.map((m) => {
            const loaded = local.loaded.find((l) => l.id === m.id);
            const speed = speedLabel(m.tps, m.measured);
            return (
              <li key={m.id}>
                <div className="skill-main">
                  <span className="skill-name">
                    {bareName(m.id)}
                    {loaded && <span className="badge is-live">{T.inMemory}</span>}
                  </span>
                  {loaded && <span className="skill-desc">{T.loadedAs(loaded.gpu_layers, loaded.layers, loaded.ctx)}</span>}
                  <span className="option-meta">
                    {(m.quant || quantOf(m.id)) && <span className="tag">{m.quant || quantOf(m.id)}</span>}
                    {m.size ? <span className="tag">{gb(m.size)}</span> : null}
                    {speed && (
                      <span className="tag" title={speedHint(m.measured)}>
                        {speed}
                      </span>
                    )}
                    {m.embedding && <span className="tag">{T.embeds}</span>}
                    {m.thinks && <span className="tag">{T.thinks}</span>}
                    {m.vision && <span className="tag">{T.vision}</span>}
                  </span>
                </div>
                <div className="skill-actions">
                  {loaded && (
                    <button className="ghost" title={T.unloadHint} disabled={loaded.busy} onClick={() => change(() => api.unloadModel(m.id))}>
                      {T.unload}
                    </button>
                  )}
                  <button className="ghost" disabled={loaded?.busy} onClick={() => remove(m)}>
                    {T.remove}
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      </section>

      {downloads.length > 0 && (
        <section aria-label="Downloads">
          <h3 className="models-head">Downloads</h3>
          <ul className="skill-list">
            {downloads.map((d) => (
              <li key={keyOf(d)} className="download">
                <div className="skill-main">
                  <span className="skill-name">{d.file}</span>
                  <span className="skill-desc">
                    {d.error ? d.error : d.finished ? T.ready : d.verifying ? T.verifying : T.of(gb(d.done), gb(d.total))}
                  </span>
                  {active(d) && (
                    <progress max={d.total || 1} value={d.done} aria-label={T.downloading(d.file)} />
                  )}
                </div>
                <div className="skill-actions">
                  {d.error && (
                    <button className="solid" title={T.resumeHint} onClick={() => start(d.repo, d.file)}>
                      {T.resume}
                    </button>
                  )}
                  <button className="ghost" onClick={() => cancel(d)}>
                    {active(d) ? T.cancel : T.dismiss}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-label={T.recommended}>
        <h3 className="models-head">{T.forThis}</h3>
        <p className="hint">{T.forThisHint}</p>
        {!suggested && <p className="hint">{T.asking}</p>}
        {suggested?.length === 0 && <p className="hint">{T.hfDown}</p>}
        <ul className="skill-list">
          {suggested?.map((s) => (
            <li key={s.repo} className={s.best ? "is-best" : undefined}>
              <div className="skill-main">
                <span className="skill-name">
                  {s.repo.split("/")[1].replace(/-GGUF$/i, "")}
                  {s.best && <span className="badge is-live">{T.best}</span>}
                </span>
                <span className="skill-desc">{[reasonLine(s.why), s.about].filter(Boolean).join(". ")}</span>
                {s.pick && <FileTags file={s.pick} vision={s.vision} thinks={s.thinks} />}
              </div>
              <div className="skill-actions">
                {s.pick ? (
                  <DownloadButton file={s.pick} running={running(s.repo, s.pick.name)} onStart={() => start(s.repo, s.pick!.name)} />
                ) : (
                  <span className="hint">{T.noFit}</span>
                )}
              </div>
            </li>
          ))}
        </ul>
      </section>

      <section aria-label={T.search}>
        <h3 className="models-head">{T.search}</h3>
        <div className="field">
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder={T.searchPlaceholder}
            aria-label={T.searchLabel}
          />
        </div>
        <p className="hint">{T.searchHint}</p>
        {repos?.length === 0 && <p className="hint">{T.nothing}</p>}
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

/** What this machine has to run models on: the CPU, memory, each GPU and the backend lair uses. */
function HardwarePanel({ hw }: { hw: Hardware | null }) {
  if (!hw) return <p className="hw-line">{T.reading}</p>;
  return (
    <div className="hw-panel">
      <p className="hw-line">
        {hw.cpu || "CPU"}
        <span className="tag">{T.cores(hw.cores)}</span>
        {hw.ram > 0 && <span className="tag">{(hw.ram_free > 0 ? T.free(gb(hw.ram_free), gb(hw.ram)) : gb(hw.ram)) + T.ram}</span>}
        <span className={`tag${hw.backend === "cpu" ? "" : " is-on"}`}>{T.runsOn[hw.backend]}</span>
      </p>
      {hw.gpus.length === 0 && <p className="hw-line">{T.noGpu}</p>}
      {hw.gpus.map((g, i) => (
        <p className="hw-line" key={i}>
          {g.name}
          <span className="tag">{(g.free > 0 && g.free < g.total ? T.free(gb(g.free), gb(g.total)) : gb(g.total)) + T.vram}</span>
        </p>
      ))}
      {hw.wsl && hw.backend === "cpu" && <p className="hint">{T.wsl}</p>}
    </div>
  );
}

function FileTags({ file, vision, thinks }: { file: HubFile; vision?: boolean; thinks?: boolean }) {
  const quant = quantOf(file.name.replace(/(-\d{5}-of-\d{5})?\.gguf$/i, ""));
  const speed = speedLabel(file.tps);
  return (
    <span className="option-meta">
      {quant && <span className="tag">{quant}</span>}
      <span className="tag">{gb(file.size)}</span>
      {(file.parts ?? 1) > 1 && <span className="tag">{T.parts(file.parts!)}</span>}
      <span className={`tag fit-${file.fit}`} title={FIT[file.fit].hint}>
        {FIT[file.fit].label}
      </span>
      {speed && (
        <span className="tag" title={speedHint(false)}>
          {speed}
        </span>
      )}
      {thinks && <span className="tag">{T.thinks}</span>}
      {vision && <span className="tag">{T.vision}</span>}
    </span>
  );
}

function DownloadButton({ file, running, onStart }: { file: HubFile; running?: Download; onStart: () => void }) {
  if (file.installed) return <span className="badge">{T.installed}</span>;
  if (running) return <span className="hint">{Math.round((running.done / (running.total || 1)) * 100)}%</span>;
  return (
    <button className={file.fit === "no" ? "ghost" : "solid"} onClick={onStart} title={FIT[file.fit].hint}>
      {T.download}
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
  running: (repo: string, file: string) => Download | undefined;
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
        <span className="skill-desc">{repo.downloads.toLocaleString()} downloads</span>
      </button>
      {open && (
        <div className="repo-files">
          {error && <p className="error">{error}</p>}
          {!files && !error && <p className="hint">{T.listing}</p>}
          {files && models.length === 0 && <p className="hint">{T.noFiles}</p>}
          <ul className="skill-list">
            {models.map((f) => (
              <li key={f.name}>
                <div className="skill-main">
                  <span className="skill-name">
                    {f.name}
                    {f.name === files?.pick && <span className="badge is-live">{T.recommended}</span>}
                  </span>
                  <FileTags file={f} vision={vision} />
                </div>
                <div className="skill-actions">
                  <DownloadButton file={f} running={running(repo.id, f.name)} onStart={() => onStart(repo.id, f.name)} />
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </li>
  );
}
