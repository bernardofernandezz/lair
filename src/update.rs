//! `lair update`: swaps this binary for the one in the latest GitHub release.

use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

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
    if cfg!(all(target_os = "linux", target_arch = "aarch64", not(feature = "cuda"), not(feature = "vulkan"))) {
        return Ok("lair-linux-arm64-cpu");
    }
    if cfg!(all(target_os = "windows", target_arch = "x86_64", not(feature = "cuda"))) {
        return Ok(if cfg!(feature = "vulkan") { "lair-windows-x86_64-vulkan" } else { "lair-windows-x86_64-cpu" });
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
    if !is_newer(&tag, current) {
        println!("lair {current} is up to date (latest release: {tag})");
        return Ok(());
    }

    let exe = std::env::current_exe()?;
    // Through symlinks to the real file. Not on Windows, where canonical paths (`\\?\C:\...`) can't be a
    // working directory.
    let exe = if cfg!(windows) { exe } else { exe.canonicalize()? };
    // Swapping the file under brew would leave it believing the old version is installed.
    if exe.components().any(|c| c.as_os_str() == "Cellar") {
        bail!("this lair was installed with Homebrew; update it with `brew upgrade lair` ({tag} is out)");
    }
    let dir = exe.parent().context("the lair binary has no parent folder")?;
    // Next to the binary, so the final rename stays on one filesystem and replaces it atomically.
    let tmp = dir.join(".lair-update");
    println!("Updating lair {current} → {tag} ({asset})...");
    let result = install(&http, &format!("{REPO}/releases/download/{tag}"), asset, &tmp, &exe).await;
    let _ = std::fs::remove_dir_all(&tmp);
    result.with_context(|| format!("couldn't update {}", exe.display()))?;
    println!("Updated {} to {tag}. Restart any running `lair serve` to use it.", exe.display());
    Ok(())
}

async fn install(http: &reqwest::Client, base: &str, asset: &str, tmp: &Path, exe: &Path) -> Result<()> {
    let file = format!("{asset}.tar.gz");
    let sums = http.get(format!("{base}/SHA256SUMS")).send().await?.error_for_status();
    let sums = sums.context("this release has no SHA256SUMS, so the download can't be verified")?.text().await?;
    let expected = checksum_for(&sums, &file).with_context(|| format!("SHA256SUMS doesn't list {file}"))?;
    let url = format!("{base}/{file}");
    let body = http.get(&url).send().await?.error_for_status()?.bytes().await?;
    if format!("{:x}", Sha256::digest(&body)) != expected {
        bail!("{file} doesn't match its checksum; the download is corrupted or was tampered with");
    }
    std::fs::create_dir_all(tmp)?;
    let archive = tmp.join("lair.tar.gz");
    std::fs::write(&archive, &body)?;
    // By its name from inside the folder: GNU tar reads the `C:` of a Windows path as a remote host.
    if !Command::new("tar").args(["-xzf", "lair.tar.gz"]).current_dir(tmp).status()?.success() {
        bail!("tar couldn't unpack {url}");
    }
    let new = tmp.join(format!("lair{}", std::env::consts::EXE_SUFFIX));
    // Run it once before swapping, so a binary that can't start here (e.g. too old a glibc) never replaces a working one.
    let ok = Command::new(&new).arg("--version").output().is_ok_and(|o| o.status.success());
    if !ok {
        bail!("the downloaded binary doesn't run on this machine");
    }
    if cfg!(windows) {
        // Windows won't replace a program that is running, but lets it be renamed out of the way.
        let old = exe.with_extension("exe.old");
        let _ = std::fs::remove_file(&old);
        std::fs::rename(exe, &old)?;
    }
    std::fs::rename(&new, exe)?;
    Ok(())
}

/// Whether release `tag` is a later version than `current` (both `vX.Y.Z`), so a newer local build is never
/// "updated" back to an older release. A prerelease (`-rc1`) comes before the release of the same numbers.
fn is_newer(tag: &str, current: &str) -> bool {
    // The numbers, then whether it is a release: `false < true`, so `1.2.0-rc1 < 1.2.0`.
    let version = |v: &str| -> (Vec<u64>, bool) {
        let v = v.trim_start_matches('v');
        let (numbers, pre) = v.split_once('-').map_or((v, false), |(n, _)| (n, true));
        (numbers.split('.').map(|n| n.parse().unwrap_or(0)).collect(), !pre)
    };
    version(tag) > version(current)
}

/// The hash `sha256sum` printed for `file`: lines read `<hex>  <name>` (or `<hex> *<name>` in binary mode).
fn checksum_for(sums: &str, file: &str) -> Option<String> {
    sums.lines().find_map(|line| {
        let (hash, name) = line.split_once(char::is_whitespace)?;
        let name = name.trim_start().trim_start_matches('*');
        (name == file && hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()))
            .then(|| hash.to_ascii_lowercase())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_later_versions_are_updates() {
        assert!(is_newer("v1.1.0", "v1.0.0"));
        assert!(is_newer("v1.10.0", "v1.9.0"), "numeric, not string, order");
        assert!(!is_newer("v1.0.0", "v1.0.0"));
        assert!(!is_newer("v1.0.0", "v1.1.0"), "a newer local build isn't downgraded");
        assert!(is_newer("v2.0.0-rc1", "v1.0.0"));
        assert!(is_newer("v1.2.0", "v1.2.0-rc1"), "a release follows its own prerelease");
        assert!(!is_newer("v1.2.0-rc1", "v1.2.0"));
    }

    #[test]
    fn checksums_are_found_by_exact_file_name() {
        let a = "a".repeat(64);
        let b = "B".repeat(64);
        let sums = format!("{a}  lair-linux-x86_64-cpu.tar.gz\n{b} *lair-macos-arm64.tar.gz\nnot a line\n");
        assert_eq!(checksum_for(&sums, "lair-linux-x86_64-cpu.tar.gz"), Some(a));
        assert_eq!(checksum_for(&sums, "lair-macos-arm64.tar.gz"), Some("b".repeat(64)));
        assert_eq!(checksum_for(&sums, "lair-linux-x86_64-vulkan.tar.gz"), None);
        assert_eq!(checksum_for(&sums, "lair-windows-x86_64-cpu.tar.gz"), None, "another system's build");
        assert_eq!(checksum_for(&sums, "cpu.tar.gz"), None, "a suffix is not a match");
        assert_eq!(checksum_for("abc  x.tar.gz", "x.tar.gz"), None, "a short hash is malformed");
    }
}
