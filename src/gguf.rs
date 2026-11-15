//! How GGUF model files are named and found on disk, and what their metadata says.

use std::path::{Path, PathBuf};

/// What a GGUF says about itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Meta {
    /// `general.architecture`: qwen3, gemma3, llama, bert…
    pub arch: String,
    /// Transformer blocks; llama.cpp offloads these plus the output layer.
    pub layers: u32,
    /// The context it was trained with.
    pub ctx_train: u32,
    pub params: u64,
    /// `Q4_K_M` and the like; empty when the file doesn't say.
    pub quant: String,
    /// An embedding model (it declares a pooling type): turns text into vectors, doesn't chat.
    pub embedding: bool,
}

/// `general.file_type` (llama.cpp's `llama_ftype`) as the name people know.
fn quant_name(ftype: u32) -> &'static str {
    match ftype {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        _ => "",
    }
}

/// Reads [`Meta`] through `get`, which looks a metadata key up.
pub fn meta_from(get: impl Fn(&str) -> Option<String>, params: u64) -> Meta {
    let arch = get("general.architecture").unwrap_or_default();
    let num = |key: &str| get(&format!("{arch}.{key}")).and_then(|v| v.parse().ok()).unwrap_or(0);
    Meta {
        layers: num("block_count"),
        ctx_train: num("context_length"),
        params,
        quant: get("general.file_type").and_then(|v| v.parse().ok()).map_or("", quant_name).to_owned(),
        embedding: get(&format!("{arch}.pooling_type")).is_some(),
        arch,
    }
}

/// `name` as a part of a split model: its base name, part number and part count
/// (`m-00001-of-00003.gguf` is part 1 of 3 of `m`). llama.cpp loads every part from the first.
pub fn shard_of(name: &str) -> Option<(&str, u32, u32)> {
    let (rest, total) = name.strip_suffix(".gguf")?.rsplit_once("-of-")?;
    let (base, index) = rest.rsplit_once('-')?;
    let digits = |s: &str| s.len() == 5 && s.bytes().all(|c| c.is_ascii_digit());
    (digits(index) && digits(total) && !base.is_empty()).then(|| (base, index.parse().unwrap(), total.parse().unwrap()))
}

/// Every file `path` is made of: itself, or each part of the split model it is the first part of.
pub fn parts(path: &Path) -> Vec<PathBuf> {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    match shard_of(&name) {
        Some((base, _, total)) => (1..=total).map(|i| path.with_file_name(shard_name(base, i, total))).collect(),
        None => vec![path.to_owned()],
    }
}

/// The file name of part `index` of `total` of the split model `base`.
pub fn shard_name(base: &str, index: u32, total: u32) -> String {
    format!("{base}-{index:05}-of-{total:05}.gguf")
}

pub fn collect_gguf(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            collect_gguf(&entry?.path(), out)?;
        }
    } else if path.extension().is_some_and(|e| e == "gguf") {
        out.push(path.to_owned());
    }
    Ok(())
}

/// Projector files ship next to the weights as `mmproj-*.gguf` (or `*-mmproj-*.gguf`).
pub fn is_mmproj(file_name: &str) -> bool {
    file_name.to_ascii_lowercase().contains("mmproj")
}

/// The id a model file is listed by: its name without `.gguf`, or a split model's base name. `None` for
/// the later parts of a split model, which is listed once, by its first.
pub fn model_id(file_name: &str) -> Option<&str> {
    match shard_of(file_name) {
        Some((base, 1, _)) => Some(base),
        Some(_) => None,
        None => Some(file_name.trim_end_matches(".gguf")),
    }
}

/// The projector for `model`: in its folder, named after it, or the only one there beside a lone model.
pub fn projector_for(model: &Path, models: &[PathBuf], projectors: &[PathBuf]) -> Option<PathBuf> {
    let dir = model.parent();
    let here: Vec<&PathBuf> = projectors.iter().filter(|p| p.parent() == dir).collect();
    let stem = model.file_stem()?.to_string_lossy().to_ascii_lowercase();
    // `Qwen2.5-VL-3B-Q4_K_M` → `qwen2.5-vl-3b`: projectors don't carry the weights' quantization.
    let base = stem.rsplit_once(['-', '.']).map_or(stem.as_str(), |(b, q)| {
        if q.starts_with('q') || q.starts_with("f16") || q.starts_with("bf16") { b } else { stem.as_str() }
    });
    let named =
        here.iter().find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains(base)));
    let alone = models.iter().filter(|m| m.parent() == dir).count() == 1 && here.len() == 1;
    named.or(alone.then(|| &here[0])).map(|p| (*p).clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projector_pairs_by_name_or_lone_folder() {
        let p = |s: &str| PathBuf::from(s);
        let models = [p("/m/Qwen2.5-VL-3B-Q4_K_M.gguf"), p("/m/Qwen3-4B-Q4_K_M.gguf"), p("/v/gemma.gguf")];
        let projs = [p("/m/mmproj-Qwen2.5-VL-3B-f16.gguf"), p("/v/mmproj-model-f16.gguf")];
        assert_eq!(projector_for(&models[0], &models, &projs), Some(projs[0].clone()));
        assert_eq!(projector_for(&models[1], &models, &projs), None, "a text model in a shared folder stays text-only");
        assert_eq!(
            projector_for(&models[2], &models, &projs),
            Some(projs[1].clone()),
            "lone model takes the lone projector"
        );
        assert!(is_mmproj("mmproj-Qwen2.5-VL-3B-f16.gguf") && !is_mmproj("Qwen2.5-VL-3B-Q4_K_M.gguf"));
    }

    #[test]
    fn gguf_metadata_is_read_by_architecture() {
        let get = |key: &str| {
            let v = match key {
                "general.architecture" => "qwen3",
                "qwen3.block_count" => "36",
                "qwen3.context_length" => "40960",
                "general.file_type" => "15",
                _ => return None,
            };
            Some(v.to_owned())
        };
        let m = meta_from(get, 4_000_000_000);
        assert_eq!((m.arch.as_str(), m.layers, m.ctx_train, m.quant.as_str()), ("qwen3", 36, 40960, "Q4_K_M"));
        assert!(!m.embedding);
        let bert = meta_from(|k| (k == "general.architecture" || k == "bert.pooling_type").then(|| "bert".into()), 0);
        assert!(bert.embedding && bert.quant.is_empty());
    }

    #[test]
    fn split_models_are_recognized() {
        assert_eq!(shard_of("Qwen3-235B-Q4_K_M-00001-of-00003.gguf"), Some(("Qwen3-235B-Q4_K_M", 1, 3)));
        assert_eq!(shard_of("Qwen3-8B-Q4_K_M.gguf"), None);
        assert_eq!(shard_of("model-of-00003.gguf"), None);
        assert_eq!(shard_name("m", 2, 3), "m-00002-of-00003.gguf");
        assert_eq!(model_id("m-00001-of-00003.gguf"), Some("m"));
        assert_eq!(model_id("m-00002-of-00003.gguf"), None, "listed by its first part");
        assert_eq!(model_id("qwen.gguf"), Some("qwen"));
    }
}
