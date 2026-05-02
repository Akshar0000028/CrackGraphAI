# CrackGraphAI Production Deployment Script for Windows
# Run as Administrator for best results

param(
    [Parameter()]
    [ValidateSet("deploy", "start", "stop", "restart", "update", "logs", "status", "build", "verify")]
    [string]$Action = "deploy",
    
    [switch]$SkipPrerequisites,
    [switch]$SkipBuild,
    [switch]$Gpu
)

$ErrorActionPreference = "Stop"

# Colors
$Green = "Green"
$Yellow = "Yellow"
$Red = "Red"
$Cyan = "Cyan"

# Paths
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$ComposeFile = Join-Path $ProjectDir "docker-compose.prod.yml"
$EnvFile = Join-Path $ProjectDir ".env"

function Write-Info($message) {
    Write-Host "[INFO] $message" -ForegroundColor $Green
}

function Write-Warn($message) {
    Write-Host "[WARN] $message" -ForegroundColor $Yellow
}

function Write-Error($message) {
    Write-Host "[ERROR] $message" -ForegroundColor $Red
}

function Write-Step($message) {
    Write-Host "`n>> $message" -ForegroundColor $Cyan
}

function Test-Prerequisites {
    Write-Step "Checking Prerequisites"
    
    # Check Docker
    try {
        $dockerVersion = docker --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Docker found: $dockerVersion"
        } else {
            throw "Docker not found"
        }
    } catch {
        Write-Error "Docker is not installed or not in PATH"
        Write-Host "Please install Docker Desktop: https://docs.docker.com/desktop/install/windows-install/"
        exit 1
    }
    
    # Check Docker Compose
    try {
        $composeVersion = docker compose version 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Docker Compose found: $composeVersion"
        } else {
            throw "Docker Compose not found"
        }
    } catch {
        Write-Error "Docker Compose not found"
        exit 1
    }
    
    # Check NVIDIA Docker (optional)
    if ($Gpu) {
        try {
            $nvidiaInfo = docker info 2>$null | Select-String "nvidia"
            if ($nvidiaInfo) {
                Write-Info "NVIDIA Docker runtime detected"
            } else {
                Write-Warn "NVIDIA Docker runtime not detected. GPU support may not be available."
                Write-Host "Install NVIDIA Container Toolkit for GPU support: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            }
        } catch {
            Write-Warn "Could not check NVIDIA Docker runtime"
        }
    }
    
    Write-Info "Prerequisites check passed"
}

function Initialize-Environment {
    Write-Step "Setting up Environment"
    
    Set-Location $ProjectDir
    
    # Create .env if it doesn't exist
    if (-not (Test-Path $EnvFile)) {
        if (Test-Path "$EnvFile.example") {
            Write-Warn ".env file not found. Creating from example..."
            Copy-Item "$EnvFile.example" $EnvFile
            Write-Warn "Please update .env with your production settings!"
        } else {
            Write-Warn "Creating minimal .env file..."
            @"
# CrackGraphAI Production Environment
API_KEY=change-me-in-production-$(Get-Random -Maximum 9999)
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
CONFIG_PATH=configs/config.yaml
WEIGHTS_PATH=checkpoints/best_hybrid_segformer.pth
CUDA_VISIBLE_DEVICES=0
CORS_ORIGINS=*
GRAFANA_PASSWORD=admin
USE_TTA=true
CACHE_TTL=3600
"@ | Out-File -FilePath $EnvFile -Encoding UTF8
        }
    }
    
    # Create required directories
    $dirs = @("outputs", ".cache", "logs", "optimized_models", "nginx/ssl")
    foreach ($dir in $dirs) {
        $path = Join-Path $ProjectDir $dir
        if (-not (Test-Path $path)) {
            New-Item -ItemType Directory -Path $path -Force | Out-Null
            Write-Info "Created directory: $dir"
        }
    }
    
    # Check for model weights
    $modelPath = Join-Path $ProjectDir "checkpoints\best_hybrid_segformer.pth"
    if (-not (Test-Path $modelPath)) {
        Write-Warn "Model weights not found at checkpoints/best_hybrid_segformer.pth"
        Write-Host "Please ensure your trained model is in place."
        Write-Host "Expected: $modelPath"
    } else {
        $size = (Get-Item $modelPath).Length / 1MB
        Write-Info "Model found: {0:N1} MB" -f $size
    }
}

function Build-Images {
    Write-Step "Building Docker Images"
    
    Set-Location $ProjectDir
    
    $buildArgs = @("-f", $ComposeFile, "build")
    if (-not $SkipBuild) {
        $buildArgs += "--no-cache"
    }
    
    & docker compose @buildArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker build failed"
        exit 1
    }
    
    Write-Info "Build complete"
}

function Start-Services {
    Write-Step "Starting Production Services"
    
    Set-Location $ProjectDir
    
    $upArgs = @("-f", $ComposeFile, "up", "-d")
    if ($Gpu) {
        $upArgs += "--build"
    }
    
    & docker compose @upArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to start services"
        exit 1
    }
    
    Write-Info "Services started. Checking health..."
    
    # Wait and check health
    $maxAttempts = 30
    $attempt = 0
    $healthy = $false
    
    while ($attempt -lt $maxAttempts -and -not $healthy) {
        Start-Sleep -Seconds 2
        $attempt++
        Write-Host -NoNewline "."
        
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                $healthy = $true
            }
        } catch {
            # Continue waiting
        }
    }
    
    Write-Host ""
    
    if ($healthy) {
        Write-Info "API is healthy and ready!"
    } else {
        Write-Warn "API health check timed out. Check logs with: docker compose -f docker-compose.prod.yml logs api"
    }
    
    Show-Status
}

function Stop-Services {
    Write-Step "Stopping Services"
    
    Set-Location $ProjectDir
    & docker compose -f $ComposeFile down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Info "Services stopped"
    } else {
        Write-Error "Failed to stop services"
    }
}

function Update-Services {
    Write-Step "Updating Services (Zero-Downtime)"
    
    Set-Location $ProjectDir
    & docker compose -f $ComposeFile pull
    & docker compose -f $ComposeFile up -d --force-recreate
    
    Write-Info "Services updated"
}

function Show-Logs {
    Write-Step "Showing Logs (Ctrl+C to exit)"
    
    Set-Location $ProjectDir
    & docker compose -f $ComposeFile logs -f api
}

function Show-Status {
    Write-Step "Service Status"
    
    Set-Location $ProjectDir
    & docker compose -f $ComposeFile ps
    
    Write-Host ""
    Write-Step "Health Check"
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10
        $health = $response.Content | ConvertFrom-Json
        Write-Info "Status: $($health.status)"
        Write-Info "Model Loaded: $($health.model_loaded)"
        Write-Info "Device: $($health.device)"
        Write-Info "Version: $($health.version)"
        Write-Info "Circuit Breaker: $($health.circuit_breaker)"
    } catch {
        Write-Error "Health check failed: $_"
    }
}

function Test-Production {
    Write-Step "Running Production Verification Tests"
    
    Set-Location $ProjectDir
    
    # Check if services are running
    $containers = docker compose -f $ComposeFile ps -q 2>$null
    if (-not $containers) {
        Write-Error "No services running. Please start services first with: .\scripts\deploy-windows.ps1 -Action start"
        return
    }
    
    # Run verification
    $testScript = Join-Path $ProjectDir "scripts\verify_production.py"
    if (Test-Path $testScript) {
        & python $testScript
    } else {
        Write-Warn "Verification script not found. Running basic checks..."
        
        # Basic health check
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
            Write-Info "Health endpoint: OK"
        } catch {
            Write-Error "Health endpoint failed: $_"
        }
        
        # Check model loading
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/ready" -UseBasicParsing
            Write-Info "Ready endpoint: OK"
        } catch {
            Write-Error "Ready endpoint failed: $_"
        }
    }
}

function Invoke-QuickInference {
    Write-Step "Quick Inference Test"
    
    $testScript = Join-Path $ProjectDir "scripts\quick_inference.py"
    if (Test-Path $testScript) {
        & python $testScript --help
    } else {
        Write-Warn "Quick inference script not found"
    }
}

# Main execution
switch ($Action) {
    "deploy" {
        if (-not $SkipPrerequisites) { Test-Prerequisites }
        Initialize-Environment
        if (-not $SkipBuild) { Build-Images }
        Start-Services
        Show-Status
        Write-Host "`n========================================" -ForegroundColor $Green
        Write-Host "  Deployment Complete!" -ForegroundColor $Green
        Write-Host "========================================" -ForegroundColor $Green
        Write-Host "API:          http://localhost:8000"
        Write-Host "API Docs:     http://localhost:8000/docs"
        Write-Host "Grafana:      http://localhost:3000 (admin/admin)"
        Write-Host "Prometheus:   http://localhost:9090"
        Write-Host ""
        Write-Host "Test with:"
        Write-Host "  curl http://localhost:8000/health"
        Write-Host ""
    }
    "start" { Start-Services }
    "stop" { Stop-Services }
    "restart" { Stop-Services; Start-Services }
    "update" { Update-Services }
    "logs" { Show-Logs }
    "status" { Show-Status }
    "build" { Build-Images }
    "verify" { Test-Production }
}
