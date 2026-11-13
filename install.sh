#!/bin/sh
# Installs lair into ~/.local/bin (no root needed):
#   curl -fsSL https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.sh | sh
# LAIR_INSTALL_DIR changes where it goes; LAIR_VARIANT=cpu|vulkan skips GPU detection on Linux.
# Native Windows has its own installer, install.ps1.
set -eu

repo=bernardofernandezz/lair
dir=${LAIR_INSTALL_DIR:-$HOME/.local/bin}

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) asset=lair-macos-arm64 ;;
  Linux-aarch64 | Linux-arm64) asset=lair-linux-arm64-cpu ;;
  Linux-x86_64)
    variant=${LAIR_VARIANT:-}
    if [ -z "$variant" ] && grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
      # Vulkan inside WSL is emulated and slower than the CPU; the real GPU is reached from Windows itself.
      variant=cpu
      echo "WSL detected: installing the CPU build. To use your GPU, install lair on Windows itself (install.ps1)"
      echo "or run the CUDA Docker image. See https://github.com/$repo#windows"
    elif [ -z "$variant" ]; then
      variant=cpu
      # The Vulkan build runs on any GPU with a driver installed, and needs glibc 2.39+ (built on Ubuntu 24.04).
      glibc=$(ldd --version 2>/dev/null | head -n1 | grep -o '[0-9]*\.[0-9]*$' || echo 0.0)
      if ldconfig -p 2>/dev/null | grep -q 'libvulkan\.so\.1'; then
        if [ "${glibc%.*}" -gt 2 ] || { [ "${glibc%.*}" -eq 2 ] && [ "${glibc#*.}" -ge 39 ]; }; then
          variant=vulkan
        else
          echo "GPU found, but glibc $glibc is older than 2.39: installing the CPU build."
        fi
      fi
    fi
    asset=lair-linux-x86_64-$variant
    ;;
  *)
    echo "No prebuilt lair for $(uname -s) $(uname -m). See https://github.com/$repo#build-from-source" >&2
    exit 1
    ;;
esac

base=https://github.com/$repo/releases/latest/download
echo "Downloading $asset..."
mkdir -p "$dir"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar "$base/$asset.tar.gz" -o "$tmp/$asset.tar.gz" || {
  echo "Couldn't download $asset from the latest release." >&2
  echo "Check https://github.com/$repo/releases, or build from source." >&2
  exit 1
}

# Every release publishes SHA256SUMS; without it, or on a mismatch, nothing is installed.
curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS" || {
  echo "The latest release has no SHA256SUMS, so the download can't be verified. Nothing was installed." >&2
  exit 1
}
grep "  $asset.tar.gz\$" "$tmp/SHA256SUMS" > "$tmp/expected" || {
  echo "SHA256SUMS doesn't list $asset.tar.gz. Nothing was installed." >&2
  exit 1
}
if command -v sha256sum >/dev/null 2>&1; then check="sha256sum"; else check="shasum -a 256"; fi
(cd "$tmp" && $check -c --status expected) || {
  echo "$asset.tar.gz doesn't match its checksum: the download is corrupted or was tampered with. Nothing was installed." >&2
  exit 1
}
tar -xzf "$tmp/$asset.tar.gz" -C "$tmp"
mv "$tmp/lair" "$dir/lair"
chmod +x "$dir/lair"

echo "Installed $dir/lair"
case ":$PATH:" in
  *":$dir:"*) ;;
  *) echo "Add it to your PATH: export PATH=\"$dir:\$PATH\"" ;;
esac
echo
echo "  lair serve     web UI and API on http://localhost:8080 (downloads a starter model the first time)"
echo "  lair run       chat in the terminal"
echo "  lair pull URL  download a .gguf model"
echo "  lair list      show your models"
echo "  lair update    switch to the latest release"
