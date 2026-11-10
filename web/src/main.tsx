import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/bricolage-grotesque/standard.css";
import "./styles.css";
import { App } from "./App";
import { paintTokens } from "./palette";
import { lang } from "./i18n";

paintTokens(document.documentElement);
document.documentElement.lang = lang === "pt" ? "pt-BR" : "en";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
