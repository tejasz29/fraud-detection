# Launches the full fraud-detection stack from one place:
#   1. Local Kafka broker (kafka_2.13-3.9.2, KRaft mode — already formatted)
#   2. Producer  -> Kafka
#   3. Consumer  -> scores -> SQLite
#   4. Dashboard -> reads SQLite
# Press Ctrl-C to tear everything down.

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$KafkaHome = "kafka_2.13-3.9.2"
$ServerStart = Join-Path $KafkaHome "bin/windows/kafka-server-start.bat"
$ServerProps = "config/kafka-server.properties"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Start-Background($Name, $Command, $Log) {
    Write-Host ">> starting $Name (log: $Log)" -ForegroundColor Cyan
    $params = @{
        FilePath = "powershell"
        ArgumentList = "-NoProfile", "-Command", $Command
        PassThru = $true
        WindowStyle = "Hidden"
    }
    if ($Log) {
        $params.RedirectStandardOutput = $Log
        $params.RedirectStandardError = $Log
    }
    return Start-Process @params
}

# 1. Local Kafka broker (no format step — kafka-logs/ already has a valid cluster.id)
Write-Host ">> starting Kafka broker (local, KRaft)" -ForegroundColor Cyan
if (-not (Test-Path $ServerStart)) {
    Write-Error "Kafka not found at $ServerStart"
    exit 1
}
$kafkaJob = Start-Background "kafka" "$ServerStart $ServerProps" (Join-Path $LogDir "kafka.log")

# Give the broker a moment to become ready.
$ready = $false
for ($i = 0; $i -lt 30; $i++) {
    try {
        $out = & "$KafkaHome/bin/windows/kafka-topics.bat" --bootstrap-server localhost:9092 --list 2>$null
        if ($LASTEXITCODE -eq 0) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 2
}
if (-not $ready) {
    Write-Warning "Kafka did not become ready in time; proceeding anyway (producer/consumer will retry)."
}

# 2-4. Python processes
$jobs = @()
$jobs += Start-Background "producer" "python -m producer --data data/creditcard.csv" (Join-Path $LogDir "producer.log")
$jobs += Start-Background "consumer" "python -m consumer" (Join-Path $LogDir "consumer.log")
$jobs += Start-Background "dashboard" "streamlit run dashboard/app.py" (Join-Path $LogDir "dashboard.log")

Write-Host "`nStack is up. Dashboard: http://localhost:8501" -ForegroundColor Green
Write-Host "Press Ctrl-C to stop everything.`n" -ForegroundColor Yellow

try {
    while ($true) { Start-Sleep -Seconds 1 }
}
finally {
    Write-Host "`n>> shutting down..." -ForegroundColor Cyan
    $jobs | ForEach-Object { if (-not $_.HasExited) { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue } }
    # Stop the Kafka broker gracefully.
    $bat = Join-Path $KafkaHome "bin/windows/kafka-server-stop.bat"
    if (Test-Path $bat) { & $bat }
    if (-not $kafkaJob.HasExited) { Stop-Process -Id $kafkaJob.Id -Force -ErrorAction SilentlyContinue }
    Write-Host ">> done." -ForegroundColor Green
}
