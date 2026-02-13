param(
    [int]$ProbeBytes = 2097152,
    [int]$ProbeTimeoutSec = 25,
    [int]$InstallRetries = 3,
    [int]$BackoffBaseSec = 8
)

$ErrorActionPreference = "Stop"

function Write-Info([string]$msg) {
    Write-Host "[INFO] $msg"
}

function Write-WarnMsg([string]$msg) {
    Write-Host "[WARN] $msg" -ForegroundColor Yellow
}

function Test-DownloadSpeed {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][int]$Bytes,
        [Parameter(Mandatory = $true)][int]$TimeoutSec
    )

    $rangeEnd = $Bytes - 1
    $metrics = & curl.exe -L --range "0-$rangeEnd" -o NUL -s -S --max-time $TimeoutSec -w "http_code=%{http_code} size=%{size_download} time=%{time_total} speed=%{speed_download}`n" $Url 2>&1
    if ($LASTEXITCODE -ne 0) {
        return [PSCustomObject]@{
            Url     = $Url
            Ok      = $false
            Status  = 0
            Bytes   = 0
            Seconds = 0
            MBps    = 0
            Error   = ($metrics | Out-String).Trim()
        }
    }

    $line = ($metrics | Select-Object -Last 1).Trim()
    if ($line -notmatch "http_code=(\d+)\s+size=(\d+)\s+time=([0-9.]+)\s+speed=([0-9.]+)") {
        return [PSCustomObject]@{
            Url     = $Url
            Ok      = $false
            Status  = 0
            Bytes   = 0
            Seconds = 0
            MBps    = 0
            Error   = "Unable to parse curl metrics: $line"
        }
    }

    $httpCode = [int]$Matches[1]
    $dlBytes = [int64]$Matches[2]
    $sec = [double]$Matches[3]
    $speedBps = [double]$Matches[4]
    $mbps = if ($sec -gt 0) { ($speedBps / 1048576.0) } else { 0.0 }
    $ok = ($httpCode -ge 200 -and $httpCode -lt 300 -and $dlBytes -gt 0 -and $mbps -gt 0)
    $errorMsg = ""
    if (-not $ok) {
        $errorMsg = "http_code=$httpCode size=$dlBytes time=$sec speed_bps=$speedBps"
    }

    return [PSCustomObject]@{
        Url     = $Url
        Ok      = $ok
        Status  = $httpCode
        Bytes   = $dlBytes
        Seconds = $sec
        MBps    = [Math]::Round($mbps, 2)
        Error   = $errorMsg
    }
}

function Stop-StalePlaywrightInstallers {
    try {
        $candidates = Get-CimInstance Win32_Process | Where-Object {
            $_.Name -in @("python.exe", "node.exe") -and $_.CommandLine -and $_.CommandLine -match "playwright" -and $_.CommandLine -match "install"
        }
        foreach ($p in $candidates) {
            Write-WarnMsg "Stopping stale installer process pid=$($p.ProcessId) name=$($p.Name)"
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-WarnMsg "Unable to inspect/stop stale installer processes: $($_.Exception.Message)"
    }
}

function Remove-StaleLockAndTemp {
    $msPlaywrightDir = Join-Path $env:LOCALAPPDATA "ms-playwright"
    $lockPath = Join-Path $msPlaywrightDir "__dirlock"
    if (Test-Path $lockPath) {
        Write-WarnMsg "Found stale lock file. Removing: $lockPath"
        Remove-Item -Recurse -Force $lockPath -ErrorAction SilentlyContinue
    }

    # 清理过旧的临时下载目录，避免堆积（不会影响正在下载的目录）
    try {
        $tempRoot = $env:TEMP
        Get-ChildItem -Path $tempRoot -Directory -Filter "playwright-download-*" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-6) } |
        ForEach-Object {
            Write-Info "Removing old temp dir: $($_.FullName)"
            Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue
        }
    }
    catch {
        Write-WarnMsg "Temp cleanup skipped: $($_.Exception.Message)"
    }
}

function Invoke-PlaywrightInstallWithRetry {
    param(
        [Parameter(Mandatory = $true)][string]$Args,
        [Parameter(Mandatory = $true)][int]$Retries,
        [Parameter(Mandatory = $true)][int]$BackoffBaseSec
    )

    for ($i = 1; $i -le $Retries; $i++) {
        Stop-StalePlaywrightInstallers
        Remove-StaleLockAndTemp

        Write-Info "Running: python -m playwright $Args (attempt $i/$Retries)"
        Clear-Variable -Name tee -ErrorAction SilentlyContinue
        python -m playwright @($Args.Split(" ")) 2>&1 | Tee-Object -Variable tee
        $code = $LASTEXITCODE
        $text = ($tee | Out-String)

        if ($code -eq 0) {
            return
        }

        $isStreamDisconnect = ($text -match "stream disconnected before completion" -or $text -match "error decoding response body")
        $isTransient = $isStreamDisconnect -or
            ($text -match "ECONNRESET" -or $text -match "timed out" -or $text -match "server closed connection" -or $text -match "size mismatch" -or $text -match "Download failure")

        if (-not $isTransient -or $i -ge $Retries) {
            throw "playwright install failed (exit=$code) after attempt $i/$Retries"
        }

        $sleepSec = [Math]::Min(300, $BackoffBaseSec * [Math]::Pow(2, ($i - 1)))
        if ($isStreamDisconnect) {
            # 用户指定的特例：出现该错误时，等待时间加长
            $sleepSec = [Math]::Min(600, $sleepSec * 2)
        }
        $jitter = Get-Random -Minimum 1 -Maximum 6
        $sleepSec = [int]($sleepSec + $jitter)
        Write-WarnMsg "Transient download error detected. Sleeping ${sleepSec}s then retrying..."
        Start-Sleep -Seconds $sleepSec
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

    # 可选镜像：
    # - 清华 TUNA 通常不提供该 Chrome for Testing 压缩包（常见为 403），但这里仍然加入探测，方便直观看到不可用原因
    # - npmmirror 提供 chrome-for-testing 的国内镜像（速度通常更快）
    $mirrorHosts = @(
        "https://npmmirror.com/mirrors/chrome-for-testing",
        "https://cdn.npmmirror.com/binaries/chrome-for-testing",
        "https://storage.googleapis.com/chrome-for-testing-public",
        "https://cdn.playwright.dev/chrome-for-testing-public",
        "https://mirrors.tuna.tsinghua.edu.cn/chrome-for-testing-public"
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

    # 由于 testUrls 是由 mirrorHosts + relativePath 直接拼接出来的，
    # 这里用字符串截取还原 host，避免路径格式差异导致误判。
    $bestHost = $best.Url.Substring(0, $best.Url.Length - $relativePath.Length - 1)
    Write-Info "Selected fastest mirror host: $bestHost"

    Stop-StalePlaywrightInstallers
    Remove-StaleLockAndTemp

    # Playwright 下载 socket 超时（默认 30s 对慢网络不友好）
    $env:PLAYWRIGHT_DOWNLOAD_CONNECTION_TIMEOUT = "300000"

    # 让 ffmpeg/winldd 等小组件也走国内镜像（npmmirror）
    $env:PLAYWRIGHT_DOWNLOAD_HOST = "https://cdn.npmmirror.com/binaries/playwright"

    $env:PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST = $bestHost

    Write-Info "Installing chromium with --no-shell using selected mirror..."
    try {
        Invoke-PlaywrightInstallWithRetry -Args "install chromium --no-shell" -Retries $InstallRetries -BackoffBaseSec $BackoffBaseSec
    }
    catch {
        Write-WarnMsg "Mirror install failed, retrying once with default host..."
        Remove-Item Env:\PLAYWRIGHT_CHROMIUM_DOWNLOAD_HOST -ErrorAction SilentlyContinue
        Invoke-PlaywrightInstallWithRetry -Args "install chromium --no-shell" -Retries 1 -BackoffBaseSec $BackoffBaseSec
    }

    Write-Info "Chromium installation completed."
}
finally {
    # no-op
}

