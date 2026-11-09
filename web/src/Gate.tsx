import { useState } from "react";

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
      <label htmlFor="token">Este servidor pede um token de acesso.</label>
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
          Entrar
        </button>
      </div>
      {error && (
        <p key={attempt} id="token-error" className="error" role="alert">
          {error}
        </p>
      )}
      <p className="hint">
        É o valor passado em <code>--token</code> ou <code>LAIR_TOKEN</code> ao iniciar o servidor.
      </p>
    </form>
  );
}
