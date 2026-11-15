//! Sorting a streamed generation into reasoning and answer, and cutting it at stop sequences.

use super::{CLOSE, OPEN};
use crate::generation::Part;
use crate::util::partial_suffix;

/// Cuts streamed text before the first stop sequence, holding back what could be the start of one.
pub(super) struct StopCut {
    stops: Vec<String>,
    pending: String,
}

impl StopCut {
    pub(super) fn new(stops: &[String]) -> Self {
        Self { stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(), pending: String::new() }
    }

    /// The text safe to show, and whether a stop sequence was reached (nothing follows then).
    pub(super) fn push(&mut self, text: &str) -> (String, bool) {
        if self.stops.is_empty() {
            return (text.to_owned(), false);
        }
        self.pending.push_str(text);
        if let Some(at) = self.stops.iter().filter_map(|s| self.pending.find(s.as_str())).min() {
            self.pending.truncate(at);
            return (std::mem::take(&mut self.pending), true);
        }
        let keep = self.stops.iter().map(|s| partial_suffix(&self.pending, s)).max().unwrap_or(0);
        let rest = self.pending.split_off(self.pending.len() - keep);
        (std::mem::replace(&mut self.pending, rest), false)
    }

    /// What was held back and turned out not to be a stop sequence.
    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum State {
    /// Nothing but whitespace yet; a `<think>` here opens the reasoning block.
    Start,
    Thinking,
    /// Just closed the block; drop the whitespace models put before the answer.
    Closed,
    Answer,
}

/// Sorts streamed text into reasoning and answer, even when a tag is split across tokens.
pub(super) struct ThinkSplit {
    state: State,
    pending: String,
}

impl ThinkSplit {
    pub(super) fn new(opened: bool) -> Self {
        Self { state: if opened { State::Thinking } else { State::Start }, pending: String::new() }
    }

    pub(super) fn thinking(&self) -> bool {
        self.state == State::Thinking
    }

    pub(super) fn push(&mut self, text: &str, emit: &mut impl FnMut(Part)) {
        self.pending.push_str(text);
        loop {
            match self.state {
                State::Start => {
                    let t = self.pending.trim_start();
                    if let Some(rest) = t.strip_prefix(OPEN) {
                        self.pending = rest.to_owned();
                        self.state = State::Thinking;
                    } else if OPEN.starts_with(t) {
                        return;
                    } else {
                        self.pending = t.to_owned();
                        self.state = State::Answer;
                    }
                }
                State::Thinking => {
                    if let Some(i) = self.pending.find(CLOSE) {
                        if i > 0 {
                            emit(Part::Thinking(&self.pending[..i]));
                        }
                        self.pending.drain(..i + CLOSE.len());
                        self.state = State::Closed;
                    } else {
                        let keep = partial_suffix(&self.pending, CLOSE);
                        let n = self.pending.len() - keep;
                        if n > 0 {
                            emit(Part::Thinking(&self.pending[..n]));
                        }
                        self.pending.drain(..n);
                        return;
                    }
                }
                State::Closed => {
                    let t = self.pending.trim_start();
                    if t.is_empty() {
                        self.pending.clear();
                        return;
                    }
                    self.pending = t.to_owned();
                    self.state = State::Answer;
                }
                State::Answer => {
                    // A stray closing tag after the block is never answer text.
                    while let Some(i) = self.pending.find(CLOSE) {
                        self.pending.replace_range(i..i + CLOSE.len(), "");
                    }
                    let n = self.pending.len() - partial_suffix(&self.pending, CLOSE);
                    if n > 0 {
                        emit(Part::Answer(&self.pending[..n]));
                    }
                    self.pending.drain(..n);
                    return;
                }
            }
        }
    }

    /// Ends the reasoning block early (the caller fed the model a closing tag).
    pub(super) fn close(&mut self, emit: &mut impl FnMut(Part)) {
        if self.state == State::Thinking {
            if !self.pending.is_empty() {
                emit(Part::Thinking(&self.pending));
            }
            self.pending.clear();
            self.state = State::Closed;
        }
    }

    pub(super) fn finish(&mut self, emit: &mut impl FnMut(Part)) {
        let rest = std::mem::take(&mut self.pending);
        match self.state {
            State::Thinking if !rest.is_empty() => emit(Part::Thinking(&rest)),
            State::Start | State::Answer if !rest.trim().is_empty() => emit(Part::Answer(&rest)),
            _ => {}
        }
    }
}

/// Tokens can split multi-byte UTF-8 characters; hold incomplete bytes until the rest arrives.
#[derive(Default)]
pub(super) struct Utf8Buf(Vec<u8>);

impl Utf8Buf {
    pub(super) fn push(&mut self, bytes: &[u8]) -> String {
        self.0.extend_from_slice(bytes);
        match std::str::from_utf8(&self.0) {
            Ok(s) => {
                let out = s.to_owned();
                self.0.clear();
                out
            }
            Err(e) if e.error_len().is_none() => {
                let n = e.valid_up_to();
                let out = String::from_utf8_lossy(&self.0[..n]).into_owned();
                self.0.drain(..n);
                out
            }
            Err(_) => String::from_utf8_lossy(&std::mem::take(&mut self.0)).into_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(opened: bool, chunks: &[&str]) -> (String, String) {
        let (mut thought, mut answer) = (String::new(), String::new());
        let mut s = ThinkSplit::new(opened);
        let mut emit = |p: Part| match p {
            Part::Thinking(t) => thought.push_str(t),
            Part::Answer(t) => answer.push_str(t),
        };
        for c in chunks {
            s.push(c, &mut emit);
        }
        s.finish(&mut emit);
        (thought, answer)
    }

    #[test]
    fn think_tags_split_across_tokens() {
        let (t, a) = split(false, &["<th", "ink>", "hm", "m</th", "ink>", "\n\n", "Oi", "!"]);
        assert_eq!((t.as_str(), a.as_str()), ("hmm", "Oi!"));
    }

    #[test]
    fn template_already_opened_the_block() {
        let (t, a) = split(true, &["plan", "</think>", " done"]);
        assert_eq!((t.as_str(), a.as_str()), ("plan", "done"));
    }

    #[test]
    fn think_tag_after_answer_started_is_text() {
        let (t, a) = split(false, &["Use ", "<think>", " tags"]);
        assert_eq!((t.as_str(), a.as_str()), ("", "Use <think> tags"));
        let (_, a) = split(false, &["Feito.</th", "ink>Pronto"]);
        assert_eq!(a, "Feito.Pronto", "a stray closing tag is dropped");
    }

    #[test]
    fn no_block_and_unfinished_tag() {
        assert_eq!(split(false, &["  Oi"]), ("".into(), "Oi".into()));
        assert_eq!(split(false, &["<thi"]), ("".into(), "<thi".into()));
        assert_eq!(split(false, &["<think>still going"]), ("still going".into(), "".into()));
    }

    #[test]
    fn utf8_split_across_tokens() {
        let mut b = Utf8Buf::default();
        let e = "é".as_bytes();
        assert_eq!(b.push(&[b'a', e[0]]), "a");
        assert_eq!(b.push(&[e[1]]), "é");
        assert_eq!(b.push(&[0xff]), "\u{fffd}");
    }

    #[test]
    fn stop_sequences_cut_the_answer_even_split_across_pieces() {
        let mut cut = StopCut::new(&["\nUser:".into(), "fim".into()]);
        let mut shown = String::new();
        let mut hit = false;
        for piece in ["Hello", "\nUs", "er", ": more", " never shown"] {
            let (text, stop) = cut.push(piece);
            shown.push_str(&text);
            if stop {
                hit = true;
                break;
            }
        }
        assert!(hit);
        assert_eq!(shown, "Hello");
        let mut cut = StopCut::new(&["é!".into()]);
        let (a, _) = cut.push("caf\u{e9}");
        assert_eq!(a, "caf", "the start of a stop sequence is held back");
        assert_eq!(cut.finish(), "\u{e9}", "and shown when it turns out not to be one");
        assert_eq!(StopCut::new(&[]).push("x"), ("x".into(), false));
    }
}
