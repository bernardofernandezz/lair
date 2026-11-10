# Installs lair into %LOCALAPPDATA%\Programs\lair (no admin needed):
#   irm https://raw.githubusercontent.com/bernardofernandezz/lair/main/install.ps1 | iex
# LAIR_INSTALL_DIR changes where it goes; LAIR_VARIANT=cpu|vulkan skips GPU detection.
& {
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue' # Windows PowerShell's progress bar makes downloads crawl.
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

    $repo = 'bernardofernandezz/lair'
    if ($env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
        throw "No prebuilt lair for Windows $env:PROCESSOR_ARCHITECTURE. See https://github.com/$repo#build-from-source"
    }
    $dir = if ($env:LAIR_INSTALL_DIR) { $env:LAIR_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA 'Programs\lair' }
    $variant = $env:LAIR_VARIANT
    if (-not $variant) {
        # GPU drivers (NVIDIA, AMD, Intel) install the Vulkan loader here; the Vulkan build can't start without it.
        $variant = if (Test-Path (Join-Path $env:SystemRoot 'System32\vulkan-1.dll')) { 'vulkan' } else { 'cpu' }
    }
    $asset = "lair-windows-x86_64-$variant"
    $file = "$asset.tar.gz"
    $base = "https://github.com/$repo/releases/latest/download"
    # bsdtar, shipped with Windows 10 and later. Not whatever `tar` is on PATH: GNU tar misreads `C:` paths.
    $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
    if (-not (Test-Path $tar)) { throw 'lair needs Windows 10 (1803) or later.' }

    $tmp = Join-Path ([IO.Path]::GetTempPath()) "lair-install-$PID"
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        Write-Host "Downloading $asset..."
        try { Invoke-WebRequest -UseBasicParsing "$base/$file" -OutFile (Join-Path $tmp $file) }
        catch { throw "Couldn't download $asset from the latest release. Check https://github.com/$repo/releases, or build from source." }

        # Every release publishes SHA256SUMS; without it, or on a mismatch, nothing is installed.
        try { Invoke-WebRequest -UseBasicParsing "$base/SHA256SUMS" -OutFile (Join-Path $tmp 'SHA256SUMS') }
        catch { throw "The latest release has no SHA256SUMS, so the download can't be verified. Nothing was installed." }
        $line = Get-Content (Join-Path $tmp 'SHA256SUMS') | Where-Object { $_ -match "^[0-9a-fA-F]{64} [ *]$([regex]::Escape($file))$" } | Select-Object -First 1
        if (-not $line) { throw "SHA256SUMS doesn't list $file. Nothing was installed." }
        $actual = (Get-FileHash -Algorithm SHA256 (Join-Path $tmp $file)).Hash
        if ($actual -ne $line.Substring(0, 64)) {
            throw "$file doesn't match its checksum: the download is corrupted or was tampered with. Nothing was installed."
        }

        Push-Location $tmp
        try { & $tar -xzf $file } finally { Pop-Location }
        if ($LASTEXITCODE -ne 0) { throw "tar couldn't unpack $file. Nothing was installed." }
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        # A running lair.exe can't be overwritten; stop `lair serve` first if this fails.
        Move-Item -Force (Join-Path $tmp 'lair.exe') (Join-Path $dir 'lair.exe')
    }
    finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }

    Write-Host "Installed $dir\lair.exe"
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    if (($userPath -split ';') -notcontains $dir) {
        [Environment]::SetEnvironmentVariable('Path', (($userPath, $dir | Where-Object { $_ }) -join ';'), 'User')
        $env:Path = "$env:Path;$dir"
        Write-Host "Added it to your PATH (new terminals pick it up)."
    }
    Write-Host ''
    Write-Host '  lair serve     web UI and API on http://localhost:8080 (downloads a starter model the first time)'
    Write-Host '  lair run       chat in the terminal'
    Write-Host '  lair pull URL  download a .gguf model'
    Write-Host '  lair list      show your models'
    Write-Host '  lair update    switch to the latest release'
}
