//! How fast models run on this machine: measured for the ones that ran, estimated for the rest from
//! those measurements, and used to pick a model when a request asks for `auto`.
//!
//! Generating a token reads every weight once, so speed is bound by how fast memory moves:
//! `seconds per token = size × (f / K_gpu + (1 − f) / K_cpu)`, with `f` the share of layers on the
//! GPU and each `K` the bytes per second that side moves. A model measured fully on one side gives
//! that side's `K`; the estimate for any other model follows.
//!
//! ponytail: one constant per side. It is off for models that read a fraction of their weights per
//! token (mixture-of-experts run faster) and for tiny ones (overhead dominates). Fit a term per
//! architecture from the recorded samples if the estimates need to be tighter.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Every measurement, by model id. Kept in the store, so estimates survive restarts.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct Bench {
    pub samples: BTreeMap<String, Sample>,
}

#[derive(Clone, Copy, Debug, PartialEq, Deserialize, Serialize)]
pub struct Sample {
    /// The model's size on disk.
    pub size: u64,
    /// Share of its layers that ran on the GPU, 0 to 1.
    pub offload: f32,
    /// Generated tokens per second, averaged over `n` generations with recent ones weighing more.
    pub tps: f32,
    pub n: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    pub tps: f32,
    /// This very model ran here with this offload; otherwise the speed comes from other models.
    pub measured: bool,
}

/// Shares of layers on the GPU closer than this count as the same setup.
const SAME_OFFLOAD: f32 = 0.05;

fn median(mut v: Vec<f64>) -> Option<f64> {
    v.sort_by(f64::total_cmp);
    (!v.is_empty()).then(|| v[v.len() / 2])
}

impl Bench {
    /// Adds one generation's speed. A model that now runs with a different offload starts over.
    pub fn record(&mut self, id: &str, size: u64, offload: f32, tps: f32) {
        if !(tps.is_finite() && tps > 0.0) {
            return;
        }
        let fresh = Sample { size, offload, tps, n: 1 };
        self.samples
            .entry(id.to_owned())
            .and_modify(|s| {
                *s = if s.size == size && (s.offload - offload).abs() < SAME_OFFLOAD {
                    Sample { tps: s.tps * 0.7 + tps * 0.3, n: s.n.saturating_add(1), ..*s }
                } else {
                    fresh
                }
            })
            .or_insert(fresh);
    }

    /// Bytes per second each side moves: the median over the models that ran entirely on it.
    fn constants(&self) -> (Option<f64>, Option<f64>) {
        let side = |on_gpu: bool| {
            median(
                self.samples
                    .values()
                    .filter(|s| if on_gpu { s.offload > 1.0 - SAME_OFFLOAD } else { s.offload < SAME_OFFLOAD })
                    .map(|s| s.size as f64 * f64::from(s.tps))
                    .collect(),
            )
        };
        (side(true), side(false))
    }

    /// The speed of a model of `size` bytes with `offload` of its layers on the GPU. `id` names an
    /// installed model, whose own measurement wins. `None` when nothing measured can tell.
    pub fn predict(&self, id: Option<&str>, size: u64, offload: f32) -> Option<Prediction> {
        if let Some(s) = id.and_then(|id| self.samples.get(id))
            && s.size == size
            && (s.offload - offload).abs() < SAME_OFFLOAD
        {
            return Some(Prediction { tps: s.tps, measured: true });
        }
        let (gpu, cpu) = self.constants();
        let f = f64::from(offload.clamp(0.0, 1.0));
        // A side that carries (almost) nothing needs no constant.
        let per_byte = match (gpu, cpu) {
            (Some(g), Some(c)) => f / g + (1.0 - f) / c,
            (Some(g), None) if f > 0.95 => 1.0 / g,
            (None, Some(c)) if f < 0.05 => 1.0 / c,
            _ => return None,
        };
        let tps = 1.0 / (size.max(1) as f64 * per_byte);
        Some(Prediction { tps: tps as f32, measured: false })
    }
}

/// A model `auto` may pick.
#[derive(Clone, Debug, Default)]
pub struct Cand {
    pub id: String,
    /// Runs on this machine.
    pub local: bool,
    pub vision: bool,
    pub size: u64,
    /// Measured or estimated tokens per second; 0 when unknown.
    pub tps: f32,
    pub loaded: bool,
    /// Generations running on its node right now.
    pub active: u32,
}

/// Slower than this, a chat stops feeling like one.
const FAST_ENOUGH: f32 = 10.0;

/// The model for a request that asked for `auto`, and why, in one sentence: the biggest one that is
/// fast enough right now, else the fastest; already-loaded and local models win ties. A request with
/// images only goes to a local model that sees them (image files never leave this machine).
pub fn pick(cands: &[Cand], vision: bool) -> Option<(usize, String)> {
    // A busy node shares its speed among its generations.
    let speed = |c: &Cand| c.tps / (1 + c.active) as f32;
    let usable = || cands.iter().enumerate().filter(|(_, c)| !vision || (c.vision && c.local));
    let tie = |c: &Cand| (c.loaded, c.local);
    let fast = usable()
        .filter(|(_, c)| speed(c) >= FAST_ENOUGH)
        .max_by(|(i, a), (j, b)| (a.size, tie(a)).cmp(&(b.size, tie(b))).then(j.cmp(i)));
    if let Some((i, c)) = fast {
        let seeing = if vision { " that sees images" } else { "" };
        return Some((i, format!("the biggest model{seeing} that is fast right now (~{:.0} tokens/s)", speed(c))));
    }
    let best =
        usable().max_by(|(i, a), (j, b)| speed(a).total_cmp(&speed(b)).then(tie(a).cmp(&tie(b))).then(j.cmp(i)))?;
    let (i, c) = best;
    let why = if speed(c) > 0.0 {
        format!("the fastest model available (~{:.0} tokens/s)", speed(c))
    } else if c.loaded {
        "the model already in memory; no speed is measured yet".to_owned()
    } else {
        "the first model found; no speed is measured yet".to_owned()
    };
    Some((i, why))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn one_measurement_estimates_other_models_and_more_correct_it() {
        let mut b = Bench::default();
        assert_eq!(b.predict(None, 4 * GB, 1.0), None, "nothing measured, nothing claimed");
        b.record("a", 4 * GB, 1.0, 60.0);
        assert_eq!(b.predict(Some("a"), 4 * GB, 1.0), Some(Prediction { tps: 60.0, measured: true }));
        let twice = b.predict(None, 8 * GB, 1.0).unwrap();
        assert!(!twice.measured && (twice.tps - 30.0).abs() < 0.01, "twice the weights, half the speed: {twice:?}");
        assert_eq!(b.predict(None, 8 * GB, 0.5), None, "no CPU measurement to price the CPU half with");

        b.record("a-cpu", 4 * GB, 0.0, 6.0);
        let half = b.predict(None, 4 * GB, 0.5).unwrap();
        assert!((half.tps - 1.0 / (0.5 / 60.0 + 0.5 / 6.0)).abs() < 0.01, "{half:?}");

        b.record("a", 4 * GB, 1.0, 40.0);
        let a = b.samples["a"];
        assert!(a.n == 2 && a.tps < 60.0 && a.tps > 40.0, "{a:?}");
        b.record("a", 4 * GB, 0.0, 5.0);
        assert_eq!(b.samples["a"], Sample { size: 4 * GB, offload: 0.0, tps: 5.0, n: 1 }, "a new setup starts over");
        b.record("a", 4 * GB, 0.0, f32::NAN);
        assert_eq!(b.samples["a"].n, 1);
        let back: Bench = serde_json::from_value(serde_json::to_value(&b).unwrap()).unwrap();
        assert_eq!(back, b);
    }

    fn cand(id: &str, size: u64, tps: f32) -> Cand {
        Cand { id: id.into(), local: true, size: size * GB, tps, ..Default::default() }
    }

    #[test]
    fn auto_takes_the_biggest_model_that_is_fast_enough() {
        let id = |cands: &[Cand], vision| pick(cands, vision).map(|(i, _)| cands[i].id.clone());
        let mut cands = vec![cand("small", 2, 80.0), cand("mid", 5, 30.0), cand("big", 20, 4.0)];
        assert_eq!(id(&cands, false).unwrap(), "mid");
        cands[1].active = 3;
        assert_eq!(id(&cands, false).unwrap(), "small", "a busy node is slower");
        let slow = [cand("a", 5, 3.0), cand("b", 9, 6.0)];
        assert_eq!(id(&slow, false).unwrap(), "b", "nothing is fast: the fastest");

        let unknown = [cand("first", 5, 0.0), Cand { loaded: true, ..cand("second", 9, 0.0) }];
        let (i, why) = pick(&unknown, false).unwrap();
        assert!(unknown[i].id == "second" && why.contains("already in memory"), "{why}");
        assert_eq!(id(&[cand("first", 5, 0.0), cand("second", 9, 0.0)], false).unwrap(), "first");

        let seers = [
            cand("text", 9, 50.0),
            Cand { vision: true, ..cand("eyes", 3, 20.0) },
            Cand { vision: true, local: false, ..cand("eyes@far", 9, 50.0) },
        ];
        assert_eq!(id(&seers, true).unwrap(), "eyes", "images stay on this machine");
        assert_eq!(id(&seers[..1], true), None);
        let same = [Cand { local: false, ..cand("m@far", 5, 30.0) }, cand("m", 5, 30.0)];
        assert_eq!(id(&same, false).unwrap(), "m", "local wins a tie");
    }
}
