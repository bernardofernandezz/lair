import { useState } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Api, Skill } from "../api";
import { pick } from "../i18n";

const T = pick({
  en: { edit: (n: string) => `Edit “${n}”`, newSkill: "New skill", builtinHint: "Saving makes your own copy that replaces the built-in one; deleting the copy brings the original back.", name: "Name", description: "Description", descPlaceholder: "One sentence on what it does", when: "Trigger words, comma-separated", whenPlaceholder: "pirate, arr", instructions: "Instructions for the model", editBtn: "Edit", preview: "Preview", empty: "*No instructions yet.*", saving: "Saving…", save: "Save", cancel: "Cancel" },
  pt: { edit: (n: string) => `Editar “${n}”`, newSkill: "Nova skill", builtinHint: "Salvar cria uma cópia sua que substitui a embutida; excluir a cópia traz a original de volta.", name: "Nome", description: "Descrição", descPlaceholder: "Uma frase sobre o que ela faz", when: "Palavras-gatilho, separadas por vírgula", whenPlaceholder: "pirata, arr", instructions: "Instruções para o modelo", editBtn: "Editar", preview: "Prévia", empty: "*Sem instruções ainda.*", saving: "Salvando…", save: "Salvar", cancel: "Cancelar" },
});

type Props = { api: Api; skill: Skill | null; onDone: () => void };

export function SkillEditor({ api, skill, onDone }: Props) {
  const [name, setName] = useState(skill?.name ?? "");
  const [description, setDescription] = useState(skill?.description ?? "");
  const [when, setWhen] = useState(skill?.when.join(", ") ?? "");
  const [prompt, setPrompt] = useState(skill?.prompt ?? "");
  const [preview, setPreview] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  async function save() {
    setSaving(true);
    setError(null);
    try {
      const triggers = when.split(",").map((w) => w.trim()).filter(Boolean);
      const draft = { name: name.trim(), description: description.trim(), when: triggers, prompt };
      await (skill ? api.saveSkill(skill.id, draft) : api.createSkill(draft));
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
      <h3>{skill ? T.edit(skill.name) : T.newSkill}</h3>
      {skill?.origin === "builtin" && <p className="hint">{T.builtinHint}</p>}
      <label>
        {T.name}
        <input value={name} onChange={(e) => setName(e.target.value)} required />
      </label>
      <label>
        {T.description}
        <input value={description} onChange={(e) => setDescription(e.target.value)} placeholder={T.descPlaceholder} />
      </label>
      <label>
        {T.when}
        <input value={when} onChange={(e) => setWhen(e.target.value)} placeholder={T.whenPlaceholder} required />
      </label>
      <div className="prompt-head">
        <span>{T.instructions}</span>
        <button type="button" className="ghost" onClick={() => setPreview((p) => !p)}>
          {preview ? T.editBtn : T.preview}
        </button>
      </div>
      {preview ? (
        <div className="prompt-preview answer">
          <Markdown remarkPlugins={[remarkGfm]}>{prompt || T.empty}</Markdown>
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
        <button className="solid" disabled={saving}>
          {saving ? T.saving : T.save}
        </button>
        <button type="button" className="ghost" onClick={onDone}>
          {T.cancel}
        </button>
      </div>
    </form>
  );
}
