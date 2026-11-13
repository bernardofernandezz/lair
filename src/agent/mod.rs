//! Agent mode: the model asks for tools in its answer, lair runs them and feeds the results back.
//!
//! The protocol is plain text so it works with any model, local or on a peer. lair asks for
//! `<tool_call>{"name": …, "arguments": {…}}</tool_call>` (the Hermes format many models know) and
//! also accepts the pythonic `[name(arg='value')]` calls some families emit natively (LFM, Llama 3.2).
//! Results return in a `<tool_response>` user message, since older chat templates have no `tool` role.

pub mod tools;

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Map, Value};
use tokio::sync::oneshot;

use crate::engine::partial_suffix;

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
/// Tool steps allowed in one reply before lair stops the loop, unless `--max-steps` says otherwise.
pub const MAX_STEPS: usize = 20;
/// How long a call waits for the user to approve it.
const APPROVAL_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Debug, PartialEq)]
pub struct Call {
    pub name: String,
    pub arguments: Value,
}

/// What the parser found in a piece of streamed answer text.
#[derive(Debug, PartialEq)]
pub enum Piece {
    Text(String),
    /// A complete call. `raw` is exactly what the model wrote, which is
    /// what goes back into its history so later turns see calls in the model's own format.
    Call {
        raw: String,
        parsed: Result<Call, String>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Syntax {
    Hermes,
    Python,
}

/// Splits streamed answer text into plain text and tool calls, even when a call's opening is split
/// across tokens. Text after a call is still text: the caller decides whether the reply ends there.
pub struct CallSplit {
    /// Text that starts a call. Pythonic openings name a known tool, so `[a link](…)` stays text.
    openers: Vec<(String, Syntax)>,
    pending: String,
    inside: Option<Syntax>,
}

impl CallSplit {
    pub fn new<'a>(tools: impl IntoIterator<Item = &'a str>) -> Self {
        let mut openers = vec![(OPEN.to_owned(), Syntax::Hermes)];
        openers.extend(tools.into_iter().map(|t| (format!("[{t}("), Syntax::Python)));
        Self { openers, pending: String::new(), inside: None }
    }

    pub fn push(&mut self, text: &str) -> Vec<Piece> {
        self.pending.push_str(text);
        let mut out = Vec::new();
        loop {
            match self.inside {
                Some(Syntax::Hermes) => {
                    let Some(end) = self.pending.find(CLOSE) else { return out };
                    let body = &self.pending[..end];
                    out.push(Piece::Call { raw: format!("{OPEN}{body}{CLOSE}"), parsed: parse_json(body) });
                    self.pending.drain(..end + CLOSE.len());
                    self.inside = None;
                    continue;
                }
                Some(Syntax::Python) => {
                    let Some(end) = python_end(&self.pending) else { return out };
                    let body = &self.pending[..end];
                    out.push(Piece::Call { raw: format!("[{body}]"), parsed: parse_python(body) });
                    self.pending.drain(..end);
                    if self.pending.starts_with(']') {
                        self.pending.remove(0);
                    }
                    self.inside = None;
                    continue;
                }
                None => {}
            }
            let first = self
                .openers
                .iter()
                .filter_map(|(o, s)| self.pending.find(o.as_str()).map(|i| (i, o.len(), *s)))
                .min_by_key(|&(i, len, _)| (i, len));
            match first {
                Some((start, len, syntax)) => {
                    if start > 0 {
                        out.push(Piece::Text(self.pending[..start].to_owned()));
                    }
                    self.pending.drain(..start + if syntax == Syntax::Hermes { len } else { 1 });
                    self.inside = Some(syntax);
                }
                None => {
                    let keep = self.openers.iter().map(|(o, _)| partial_suffix(&self.pending, o)).max().unwrap_or(0);
                    let n = self.pending.len() - keep;
                    if n > 0 {
                        out.push(Piece::Text(self.pending[..n].to_owned()));
                    }
                    self.pending.drain(..n);
                    return out;
                }
            }
        }
    }

    /// Called when generation ended: an unclosed call is still tried (models often stop right before closing it).
    pub fn finish(&mut self) -> Option<Piece> {
        let rest = std::mem::take(&mut self.pending);
        match self.inside.take() {
            Some(Syntax::Hermes) => {
                Some(Piece::Call { parsed: parse_json(&rest), raw: format!("{OPEN}{rest}{CLOSE}") })
            }
            Some(Syntax::Python) => {
                let body = rest.trim_end().trim_end_matches(']');
                Some(Piece::Call { parsed: parse_python(body), raw: format!("[{body}]") })
            }
            None => (!rest.is_empty()).then_some(Piece::Text(rest)),
        }
    }
}

fn parse_json(body: &str) -> Result<Call, String> {
    #[derive(Deserialize)]
    struct Raw {
        name: String,
        #[serde(default)]
        arguments: Value,
    }
    let raw: Raw =
        serde_json::from_str(body.trim()).map_err(|e| format!("the call is not valid JSON ({e}): {}", body.trim()))?;
    let arguments = if raw.arguments.is_null() { Value::Object(Default::default()) } else { raw.arguments };
    Ok(Call { name: raw.name, arguments })
}

/// Byte index just past the first top-level `name(...)` in `s`, once its closing `)` has arrived.
fn python_end(s: &str) -> Option<usize> {
    let (mut depth, mut quote, mut escaped) = (0i32, None, false);
    for (i, c) in s.char_indices() {
        match quote {
            Some(_) if escaped => escaped = false,
            Some(_) if c == '\\' => escaped = true,
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == '\'' || c == '"' => quote = Some(c),
            None if c == '(' || c == '[' || c == '{' => depth += 1,
            None if c == ')' || c == ']' || c == '}' => {
                depth -= 1;
                if depth == 0 && c == ')' {
                    return Some(i + 1);
                }
            }
            None => {}
        }
    }
    None
}

/// Parses `name(key='value', n=2, flag=True)` into a call with JSON arguments.
fn parse_python(src: &str) -> Result<Call, String> {
    let err = |why: &str| format!("invalid call ({why}): {src}");
    let open = src.find('(').ok_or_else(|| err("missing '('"))?;
    let body = src[open + 1..].trim_end().strip_suffix(')').unwrap_or(&src[open + 1..]);
    let mut p = Py { s: body.chars().collect(), i: 0 };
    let mut args = Map::new();
    loop {
        p.ws();
        if p.done() {
            break;
        }
        let key: String = p.take_while(|c| c.is_alphanumeric() || c == '_');
        p.ws();
        if key.is_empty() || !p.eat('=') {
            return Err(err("use argument=value"));
        }
        args.insert(key, p.value().map_err(|w| err(&w))?);
        p.ws();
        if !p.eat(',') && !p.done() {
            return Err(err("separate the arguments with commas"));
        }
    }
    Ok(Call { name: src[..open].trim().to_owned(), arguments: Value::Object(args) })
}

struct Py {
    s: Vec<char>,
    i: usize,
}

impl Py {
    fn done(&self) -> bool {
        self.i >= self.s.len()
    }
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }
    fn eat(&mut self, c: char) -> bool {
        let hit = self.peek() == Some(c);
        self.i += usize::from(hit);
        hit
    }
    fn ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.i += 1;
        }
    }
    fn take_while(&mut self, f: impl Fn(char) -> bool) -> String {
        let start = self.i;
        while self.peek().is_some_and(&f) {
            self.i += 1;
        }
        self.s[start..self.i].iter().collect()
    }
    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        match self.peek().ok_or("missing value")? {
            q @ ('\'' | '"') => {
                self.i += 1;
                let mut out = String::new();
                loop {
                    let c = self.peek().ok_or("string without a closing quote")?;
                    self.i += 1;
                    match c {
                        '\\' => {
                            let e = self.peek().ok_or("incomplete escape")?;
                            self.i += 1;
                            out.push(match e {
                                'n' => '\n',
                                't' => '\t',
                                other => other,
                            });
                        }
                        c if c == q => return Ok(Value::String(out)),
                        c => out.push(c),
                    }
                }
            }
            '[' => {
                self.i += 1;
                let mut items = Vec::new();
                loop {
                    self.ws();
                    if self.eat(']') {
                        return Ok(Value::Array(items));
                    }
                    items.push(self.value()?);
                    self.ws();
                    self.eat(',');
                }
            }
            _ => {
                let word = self.take_while(|c| c.is_alphanumeric() || matches!(c, '.' | '-' | '+' | '_'));
                match word.as_str() {
                    "True" | "true" => Ok(Value::Bool(true)),
                    "False" | "false" => Ok(Value::Bool(false)),
                    "None" | "null" => Ok(Value::Null),
                    w => w
                        .parse::<f64>()
                        .ok()
                        .and_then(serde_json::Number::from_f64)
                        .map(Value::Number)
                        .ok_or(format!("unknown value {w:?}")),
                }
            }
        }
    }
}

pub fn render_result(name: &str, output: &str) -> String {
    format!("<tool_response name=\"{name}\">\n{output}\n</tool_response>")
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Decision {
    pub allow: bool,
    /// Allow this tool for the rest of the conversation without asking again.
    #[serde(default)]
    pub always: bool,
}

/// Calls waiting for the user, and the tools a conversation already allowed for good.
#[derive(Default)]
pub struct Approvals {
    pending: Mutex<HashMap<String, oneshot::Sender<Decision>>>,
    always: Mutex<HashMap<String, HashSet<String>>>,
}

impl Approvals {
    pub fn allowed(&self, conv: &str, tool: &str) -> bool {
        self.always.lock().unwrap().get(conv).is_some_and(|t| t.contains(tool))
    }

    /// Waits for the user's answer to call `id`. Dropping the future (client gone) withdraws the question.
    pub async fn ask(&self, conv: &str, tool: &str, id: &str) -> Option<Decision> {
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id.to_owned(), tx);
        let _withdraw = Withdraw(self, id);
        let decision = tokio::time::timeout(APPROVAL_TIMEOUT, rx).await.ok()?.ok()?;
        if decision.allow && decision.always {
            self.always.lock().unwrap().entry(conv.to_owned()).or_default().insert(tool.to_owned());
        }
        Some(decision)
    }

    /// Delivers the user's answer; false when no call with that id is waiting.
    pub fn answer(&self, id: &str, decision: Decision) -> bool {
        self.pending.lock().unwrap().remove(id).is_some_and(|tx| tx.send(decision).is_ok())
    }
}

struct Withdraw<'a>(&'a Approvals, &'a str);

impl Drop for Withdraw<'_> {
    fn drop(&mut self) {
        self.0.pending.lock().unwrap().remove(self.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, arguments: Value) -> Result<Call, String> {
        Ok(Call { name: name.into(), arguments })
    }

    fn parsed(p: &Piece) -> &Result<Call, String> {
        match p {
            Piece::Call { parsed, .. } => parsed,
            Piece::Text(t) => panic!("expected a call, got text {t:?}"),
        }
    }

    fn run(chunks: &[&str]) -> Vec<Piece> {
        let mut s = CallSplit::new(["list_dir", "read_file", "write_file"]);
        let mut out: Vec<Piece> = chunks.iter().flat_map(|c| s.push(c)).collect();
        out.extend(s.finish());
        out
    }

    #[test]
    fn a_call_split_across_tokens_is_found_after_its_text() {
        let out = run(&[
            "Vou olhar. <to",
            "ol_call>{\"name\":\"list_dir\",",
            "\"arguments\":{\"path\":\".\"}}</tool",
            "_call>",
        ]);
        assert_eq!(out[0], Piece::Text("Vou olhar. ".into()));
        assert_eq!(parsed(&out[1]), &call("list_dir", serde_json::json!({"path": "."})));
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn plain_text_passes_and_partial_tags_are_held_until_resolved() {
        assert_eq!(run(&["Oi <to", "do"]), [Piece::Text("Oi ".into()), Piece::Text("<todo".into())]);
    }

    #[test]
    fn bad_json_and_missing_close_still_report_a_call() {
        let out = run(&["<tool_call>{name: list_dir}</tool_call>"]);
        assert!(matches!(parsed(&out[0]), Err(e) if e.contains("JSON")));
        let out = run(&["<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"a\"}}"]);
        assert!(matches!(parsed(&out[0]), Ok(c) if c.name == "read_file"));
        let out = run(&["<tool_call>{\"name\":\"list_dir\"}</tool_call>"]);
        assert!(matches!(parsed(&out[0]), Ok(c) if c.arguments.is_object()));
    }

    #[test]
    fn pythonic_calls_are_found_and_parsed() {
        let out = run(&["Ok. [list_d", "ir(path='.')]", " resto"]);
        assert_eq!(out[0], Piece::Text("Ok. ".into()));
        assert_eq!(
            out[1],
            Piece::Call {
                raw: "[list_dir(path='.')]".into(),
                parsed: call("list_dir", serde_json::json!({"path": "."}))
            }
        );
        let out = run(&[r#"[write_file(path="a b.md", content='linha 1\nit\'s (ok)', n=2, ok=True)]"#]);
        assert_eq!(
            parsed(&out[0]),
            &call(
                "write_file",
                serde_json::json!({"path": "a b.md", "content": "linha 1\nit's (ok)", "n": 2.0, "ok": true})
            )
        );
    }

    #[test]
    fn brackets_that_are_not_tool_calls_stay_text() {
        let text: String = run(&["Veja [o guia](http://x) e [list](1)."])
            .into_iter()
            .map(|p| match p {
                Piece::Text(t) => t,
                Piece::Call { raw, .. } => panic!("unexpected call {raw}"),
            })
            .collect();
        assert_eq!(text, "Veja [o guia](http://x) e [list](1).");
        assert!(parsed(&run(&["[read_file(path=)]"])[0]).is_err());
    }

    #[tokio::test]
    async fn approvals_deliver_remember_and_withdraw() {
        let a = std::sync::Arc::new(Approvals::default());
        let waiter = tokio::spawn({
            let a = a.clone();
            async move { a.ask("c", "run_command", "call1").await }
        });
        while !a.answer("call1", Decision { allow: true, always: true }) {
            tokio::task::yield_now().await;
        }
        assert!(waiter.await.unwrap().unwrap().allow);
        assert!(a.allowed("c", "run_command"));
        assert!(!a.allowed("other", "run_command"));

        let waiter = tokio::spawn({
            let a = a.clone();
            async move { a.ask("c", "write_file", "call2").await }
        });
        tokio::task::yield_now().await;
        waiter.abort();
        let _ = waiter.await;
        assert!(!a.answer("call2", Decision { allow: true, always: false }), "an abandoned question is withdrawn");
    }
}
