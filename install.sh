#!/bin/sh
# Installs lair into ~/.local/bin (no root needed):
#   curl -fsSL https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.sh | sh
# LAIR_INSTALL_DIR changes where it goes; LAIR_VARIANT=cpu|vulkan skips GPU detection on Linux.
set -eu

repo=bernardofernandezz/lair
dir=${LAIR_INSTALL_DIR:-$HOME/.local/bin}

case "$(uname -s)-$(uname -m)" in
  Darwin-arm64) asset=lair-macos-arm64 ;;
  Linux-x86_64)
    variant=${LAIR_VARIANT:-}
    if [ -z "$variant" ]; then
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

url=https://github.com/$repo/releases/latest/download/$asset.tar.gz
echo "Downloading $asset..."
mkdir -p "$dir"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fL --progress-bar "$url" -o "$tmp/lair.tar.gz" || {
  echo "Couldn't download $asset from the latest release." >&2
  echo "Check https://github.com/$repo/releases, or build from source." >&2
  exit 1
}
tar -xzf "$tmp/lair.tar.gz" -C "$tmp"
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
