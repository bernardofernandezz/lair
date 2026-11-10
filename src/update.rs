//! `lair update`: swaps this binary for the one in the latest GitHub release.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};

use crate::net;

const REPO: &str = "https://github.com/bernardofernandezz/lair";

/// The release asset this binary was built as; the names match release.yml.
fn asset() -> Result<&'static str> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        return Ok("lair-macos-arm64");
    }
    if cfg!(all(target_os = "linux", target_arch = "x86_64", not(feature = "cuda"))) {
        return Ok(if cfg!(feature = "vulkan") { "lair-linux-x86_64-vulkan" } else { "lair-linux-x86_64-cpu" });
    }
    bail!("there's no prebuilt release for this build; update it from source ({REPO}#build-from-source)")
}

pub async fn run() -> Result<()> {
    let asset = asset()?;
    let http = net::download_client();
    // /releases/latest redirects to /releases/tag/<tag>, with no API token or rate limit involved.
    let resp = http.head(format!("{REPO}/releases/latest")).send().await?.error_for_status()?;
    let tag = resp.url().path_segments().and_then(|mut s| s.next_back()).unwrap_or_default().to_owned();
    if !tag.starts_with('v') {
        bail!("couldn't find the latest release at {REPO}/releases");
    }
    let current = concat!("v", env!("CARGO_PKG_VERSION"));
    if tag == current {
        println!("lair {current} is already the latest version");
        return Ok(());
    }

    let exe = std::env::current_exe()?.canonicalize()?;
    let dir = exe.parent().context("the lair binary has no parent folder")?;
    // Next to the binary, so the final rename stays on one filesystem and replaces it atomically.
    let tmp = dir.join(".lair-update");
    println!("Updating lair {current} → {tag} ({asset})...");
    let result = install(&http, &format!("{REPO}/releases/download/{tag}/{asset}.tar.gz"), &tmp, &exe).await;
    let _ = std::fs::remove_dir_all(&tmp);
    result.with_context(|| format!("couldn't update {}", exe.display()))?;
    println!("Updated {} to {tag}. Restart any running `lair serve` to use it.", exe.display());
    Ok(())
}

async fn install(http: &reqwest::Client, url: &str, tmp: &Path, exe: &Path) -> Result<()> {
    let body = http.get(url).send().await?.error_for_status()?.bytes().await?;
    std::fs::create_dir_all(tmp)?;
    let archive = tmp.join("lair.tar.gz");
    std::fs::write(&archive, &body)?;
    if !Command::new("tar").arg("-xzf").arg(&archive).arg("-C").arg(tmp).status()?.success() {
        bail!("tar couldn't unpack {url}");
    }
    let new = tmp.join("lair");
    // Run it once before swapping, so a binary that can't start here (e.g. too old a glibc) never replaces a working one.
    let ok = Command::new(&new).arg("--version").output().is_ok_and(|o| o.status.success());
    if !ok {
        bail!("the downloaded binary doesn't run on this machine");
    }
    std::fs::rename(&new, exe)?;
    Ok(())
}
