import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Api, Skill } from "../api";

/** A skill id from its name: lowercase ASCII words joined by hyphens. */
export function slug(name: string): string {
  return name
    .normalize("NFD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 64);
}

type Props = { api: Api; skill: Skill | null; onDone: () => void };

export function SkillEditor({ api, skill, onDone }: Props) {
  const [name, setName] = useState(skill?.name ?? "");
  const [description, setDescription] = useState(skill?.description ?? "");
  const [when, setWhen] = useState(skill?.when.join(", ") ?? "");
  const [prompt, setPrompt] = useState(skill?.prompt ?? "");
  const [preview, setPreview] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const id = skill?.id ?? slug(name);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const triggers = when.split(",").map((w) => w.trim()).filter(Boolean);
      await api.saveSkill(id, { name: name.trim(), description: description.trim(), when: triggers, prompt });
      onDone();
    } catch (e) {
      setError((e as Error).message);
      setSaving(false);
    }
  }

  return (
    <form
      className="skill-editor"
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
    >
      <h3>{skill ? `Editar “${skill.name}”` : "Nova skill"}</h3>
      {skill?.origin === "builtin" && <p className="hint">Salvar cria uma cópia sua que substitui a embutida; excluir a cópia traz a original de volta.</p>}
      <label>
        Nome
        <input value={name} onChange={(e) => setName(e.target.value)} required />
      </label>
      {!skill && name && <p className="hint">Identificador: {id || "use letras ou números no nome"}</p>}
      <label>
        Descrição
        <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder="Uma frase sobre o que ela faz" />
      </label>
      <label>
        Palavras-gatilho, separadas por vírgula
        <input value={when} onChange={(e) => setWhen(e.target.value)} placeholder="pirata, arr" required />
      </label>
      <div className="prompt-head">
        <span>Instruções para o modelo</span>
        <button type="button" className="ghost" onClick={() => setPreview((p) => !p)}>
          {preview ? "Editar" : "Prévia"}
        </button>
      </div>
      {preview ? (
        <div className="prompt-preview answer">
          <Markdown remarkPlugins={[remarkGfm]}>{prompt || "*Sem instruções ainda.*"}</Markdown>
        </div>
      ) : (
        <textarea className="prompt-input" value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={10} required />
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="editor-actions">
        <button className="solid" disabled={saving || !id}>
          {saving ? "Salvando…" : "Salvar"}
        </button>
        <button type="button" className="ghost" onClick={onDone}>
          Cancelar
        </button>
      </div>
    </form>
  );
}
