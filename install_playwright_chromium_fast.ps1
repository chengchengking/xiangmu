param(
    [int]$ProbeBytes = 2097152,
    [int]$ProbeTimeoutSec = 25
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) {
    Write-Host "[INFO] $msg"
}

function Write-WarnMsg([string]$msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

$probeScriptPath = Join-Path $PSScriptRoot ".probe_mirror_speed.py"
$probeScriptLines = @(
    "import json",
    "import ssl",
    "import sys",
    "import time",
    "import urllib.request",
    "",
    "url = sys.argv[1]",
    "probe_bytes = int(sys.argv[2])",
    "timeout_sec = int(sys.argv[3])",
    "",
    "req = urllib.request.Request(",
    "    url,",
    "    headers={",
    "        'Range': f'bytes=0-{probe_bytes - 1}',",
    "        'User-Agent': 'Mozilla/5.0',",
    "    },",
    ")",
    "",
    "start = time.time()",
    "try:",
    "    with urllib.request.urlopen(req, timeout=timeout_sec, context=ssl.create_default_context()) as resp:",
    "        data = resp.read()",
    "        elapsed = max(time.time() - start, 0.001)",
    "        mbps = (len(data) / 1048576.0) / elapsed",
    "        print(json.dumps({",
    "            'ok': True,",
    "            'status': int(getattr(resp, 'status', 200)),",
    "            'bytes': len(data),",
    "            'seconds': round(elapsed, 2),",
    "            'mbps': round(mbps, 2),",
    "            'error': '',",
    "        }))",
    "except Exception as exc:",
    "    print(json.dumps({",
    "        'ok': False,",
    "        'status': 0,",
    "        'bytes': 0,",
    "        'seconds': 0,",
    "        'mbps': 0,",
    "        'error': str(exc),",
    "    }))"
)
$probeScriptLines | Set-Content -Path $probeScriptPath -Encoding ASCII

function Test-DownloadSpeed {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][int]$Bytes,
        [Parameter(Mandatory = $true)][int]$TimeoutSec
    )

    $jsonLine = python $probeScriptPath $Url $Bytes $TimeoutSec
    if ($LASTEXITCODE -ne 0) {
        return [PSCustomObject]@{
            Url     = $Url
            Ok      = $false
            Status  = 0
            Bytes   = 0
            Seconds = 0
            MBps    = 0
            Error   = "python probe exited with code $LASTEXITCODE"
        }
    }

    $probe = $jsonLine | ConvertFrom-Json
    return [PSCustomObject]@{
        Url     = $Url
        Ok      = [bool]$probe.ok
        Status  = [int]$probe.status
        Bytes   = [int]$probe.bytes
        Seconds = [double]$probe.seconds
        MBps    = [double]$probe.mbps
        Error   = [string]$probe.error
    }
}

try {
    Write-Info "Reading Playwright chromium info via dry-run..."
    $dryRunOutput = python -m playwright install --dry-run chromium --no-shell
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to run playwright dry-run."
    }

    $downloadLine = $dryRunOutput | Where-Object { $_ -match "Download url:\s+https?://.*/chrome-win64\.zip" } | Select-Object -First 1
    if (-not $downloadLine) {
        throw "Could not parse chromium download URL from dry-run output."
    }

    $defaultUrl = ($downloadLine -replace "^\s*Download url:\s+", "").Trim()
    Write-Info "Default chromium URL: $defaultUrl"

    if ($defaultUrl -notmatch "^https?://[^/]+/chrome-for-testing-public/(.+)$") {
        throw "Unexpected chromium URL format; cannot build mirror probe URLs."
    }
    $relativePath = $Matches[1]

    $mirrorHosts = @(
        "https://storage.googleapis.com/chrome-for-testing-public",
        "https://cdn.playwright.dev/chrome-for-testing-public"
    )

    $testUrls = $mirrorHosts | ForEach-Object { "$_/$relativePath" }
    Write-Info "Probing mirrors by downloading first $ProbeBytes bytes from each URL..."

    $results = @()
    foreach ($u in $testUrls) {
        $r = Test-DownloadSpeed -Url $u -Bytes $ProbeBytes -TimeoutSec $ProbeTimeoutSec
        $results += $r
        if ($r.Ok) {
            Write-Info "Probe ok: url=$($r.Url) status=$($r.Status) bytes=$($r.Bytes) time=$($r.Seconds)s speed=$($r.MBps)MB/s"
        }
        else {
            Write-WarnMsg "Probe failed: url=$($r.Url) error=$($r.Error)"
        }
    }

    $best = $results | Where-Object { $_.Ok } | Sort-Object MBps -Descending | Select-Object -First 1
    if (-not $best) {
        throw "All mirrors failed during probe."
    }

    $bestHost = $best.Url.Substring(0, $best.Url.Length - $relativePath.Length - 1)
    Write-Info "Selected fastest mirror host: $bestHost"

    $lockPath = "C:\Users\aChen\AppData\Local\ms-playwright\__dirlock"
    if (Test-Path $lockPath) {
        Write-WarnMsg "Found stale lock file. Removing: $lockPath"
        Remove-Item -Recurse -Force $lockPath
    }

    $env:HTTP_PROXY = ""
    $env:HTTPS_PROXY = ""
    $env:ALL_PROXY = ""
    $env:PIP_NO_INDEX = "0"
    $env:PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST = $bestHost

    Write-Info "Installing chromium with --no-shell using selected mirror..."
    python -m playwright install chromium --no-shell
    if ($LASTEXITCODE -ne 0) {
        Write-WarnMsg "Mirror install failed, retrying with default host..."
        Remove-Item Env:\PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST -ErrorAction SilentlyContinue
        python -m playwright install chromium --no-shell
        if ($LASTEXITCODE -ne 0) {
            throw "Default host install failed as well."
        }
    }

    Write-Info "Chromium installation completed."
}
finally {
    if (Test-Path $probeScriptPath) {
        Remove-Item -Force $probeScriptPath -ErrorAction SilentlyContinue
    }
}
