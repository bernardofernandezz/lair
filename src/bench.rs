//! `lair bench`: measures how fast a model runs here, for the estimates of every other model.

use anyhow::{Context, Result};

use crate::engine::{Engine, Tune};
use crate::generation::{Msg, Params, Reasoning};
use crate::speed::Bench;
use crate::store::Store;

/// Generates a fixed amount with the model as lair would run it, and again on the CPU alone when a GPU
/// took part, so both sides of the speed estimate get a measurement.
pub fn run(mut engine: Engine, id: &str, params: &Params, store: &Store) -> Result<()> {
    // Only what this run measures is saved, over what is stored: a running server may be saving its own.
    let mut bench = Bench::default();
    let offload = measure(&engine, id, params, &mut bench, id, "as lair runs it")?;
    if offload > 0.0 {
        engine.set_tune(Tune { gpu_layers: Some(0), ..engine.tune() });
        // Kept under its own name: one sample per model would let the CPU run replace the GPU one.
        measure(&engine, id, params, &mut bench, &format!("{id} (cpu)"), "on the CPU alone")?;
    }
    bench.save(store)?;
    println!("saved; lair now estimates other models' speed on this machine from this");
    Ok(())
}

/// Runs `id` once, records its speed under `key` and returns the share of its layers on the GPU.
fn measure(engine: &Engine, id: &str, params: &Params, bench: &mut Bench, key: &str, label: &str) -> Result<f32> {
    let msgs = [Msg {
        role: "user".into(),
        content: "Write a long, detailed essay about the history of computing.".into(),
        ..Default::default()
    }];
    let p = Params { max_tokens: 96, reasoning: Reasoning::Off, temperature: 0.0, ..params.clone() };
    eprintln!("measuring {id} {label}…");
    let done = engine.generate(id, &msgs, &p, |_| true)?;
    let l = engine.loaded().into_iter().find(|l| l.id == id).context("the model just ran")?;
    let reads = engine.info(id).map_or(l.size, |m| m.meta.read_per_token(l.size));
    bench.record(key, reads, l.offload(), done.tps());
    println!(
        "{id} {label}: {:.1} tokens/s ({} of {} layers on the GPU, context {}, loaded in {:.1} s)",
        done.tps(),
        l.gpu_layers,
        l.layers,
        l.ctx,
        done.load_us as f64 / 1e6
    );
    Ok(l.offload())
}
