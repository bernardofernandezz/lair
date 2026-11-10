import { useState } from "react";
import { pick } from "./i18n";

const T = pick({
  en: { ask: "This server needs an access token.", enter: "Sign in", hint1: "It's the value given in ", hint2: " or ", hint3: " when the server started." },
  pt: { ask: "Este servidor pede um token de acesso.", enter: "Entrar", hint1: "É o valor passado em ", hint2: " ou ", hint3: " ao iniciar o servidor." },
});

type Props = { error?: string; attempt: number; onSubmit: (token: string) => Promise<void> };

export function Gate({ error, attempt, onSubmit }: Props) {
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);

  return (
    <form
      className="center gate"
      onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        await onSubmit(token);
        setBusy(false);
      }}
    >
      <label htmlFor="token">{T.ask}</label>
      <div className="field">
        <input
          id="token"
          type="password"
          autoComplete="current-password"
          autoFocus
          value={token}
          onChange={(e) => setToken(e.target.value)}
          aria-invalid={!!error}
          aria-describedby={error ? "token-error" : undefined}
        />
        <button className="solid" disabled={!token || busy}>
          {T.enter}
        </button>
      </div>
      {error && (
        <p key={attempt} id="token-error" className="error" role="alert">
          {error}
        </p>
      )}
      <p className="hint">
        {T.hint1}
        <code>--token</code>
        {T.hint2}
        <code>LAIR_TOKEN</code>
        {T.hint3}
      </p>
    </form>
  );
}
