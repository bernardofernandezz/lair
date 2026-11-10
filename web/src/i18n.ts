/** UI language: Portuguese for pt-* browsers, English for everyone else. */
export const lang: "en" | "pt" = typeof navigator !== "undefined" && navigator.language?.toLowerCase().startsWith("pt") ? "pt" : "en";

/** Picks this browser's strings; `pt` must have exactly the keys `en` has.
 *  shortcut: no runtime switching or plurals; move to Intl/a library when a third language arrives. */
export function pick<T>(d: { en: T; pt: NoInfer<T> }): T {
  return d[lang];
}
