# MOSS-Image LoRA 학습 도구 설치 스크립트. install.bat 이 실행한다.
# 받은 파일은 이 폴더 안(tools, sd-scripts)에만 둔다. 시스템 파이썬이나 PATH 는 바꾸지 않는다.
# 다시 실행해도 된다. 이미 있는 것은 건너뛰고, 사용자가 만든 설정은 덮어쓰지 않는다.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'  # Invoke-WebRequest 진행 표시는 다운로드를 크게 늦춘다
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $PSScriptRoot
$Tools = Join-Path $Root 'tools'
$SdDir = Join-Path $Root 'sd-scripts'
$Py = Join-Path $SdDir 'venv\Scripts\python.exe'
$Uv = Join-Path $Tools 'uv\uv.exe'

# 이 조합으로 시험했다. 올릴 때는 함께 시험하고 해시도 바꾼다
$SdCommit = 'cb70b96858477e716a2096292ab8aab433277f38'  # kohya-ss/sd-scripts v0.12.3
$PythonVersion = '3.11'
$TorchPackages = @('torch==2.9.1+cu128', 'torchvision==0.24.1+cu128')
$TorchIndex = 'https://download.pytorch.org/whl/cu128'
$UvVersion = '0.13.6'
$UvSha256 = 'a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f'
$Fa2File = 'flash_attn-2.8.3+cu128torch2.9-cp311-cp311-win_amd64.whl'
$Fa2Url = 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.13/flash_attn-2.8.3%2Bcu128torch2.9-cp311-cp311-win_amd64.whl'
$Fa2Sha256 = '2c5a1835ff8cd1a801de1917aacb6f66461606ab1ff4ef90f5e7b4d361a38309'

function Step($Number, $Text) { Write-Host "`n[$Number/7] $Text" -ForegroundColor Cyan }
function Note($Text) { Write-Host "  $Text" }
function Warn($Text) { Write-Host "  주의: $Text" -ForegroundColor Yellow }

function Get-Verified($Url, $Path, $Sha256) {
    if ((Test-Path -LiteralPath $Path) -and (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ieq $Sha256) { return }
    Invoke-WebRequest -Uri $Url -OutFile $Path -UseBasicParsing
    $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
    if ($hash -ine $Sha256) {
        Remove-Item -LiteralPath $Path -Force
        throw "받은 파일의 해시가 다릅니다: $Url"
    }
}

function Run($Exe, [string[]]$Arguments) {
    # 윈도우 PowerShell 5.1 은 외부 프로그램의 stderr 를 오류로 취급할 수 있어, 성공 여부는 종료 코드로만 본다
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments
    $code = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($code -ne 0) { throw "실패(종료 코드 $code): $Exe $($Arguments -join ' ')" }
}

try {
    Write-Host 'MOSS-Image LoRA 학습 도구 설치' -ForegroundColor Green
    Write-Host "설치 위치: $Root"

    Step 1 '환경 확인'
    if (-not [Environment]::Is64BitOperatingSystem) { throw '64비트 Windows 가 필요합니다.' }
    New-Item -ItemType Directory -Force -Path $Tools | Out-Null
    $freeGb = [math]::Floor((Get-Item -LiteralPath $Root).PSDrive.Free / 1GB)
    Note "디스크 여유 공간: ${freeGb}GB"
    if ($freeGb -lt 15) { Warn '설치에 약 6GB 가 필요합니다. 모델 파일은 약 27GB 가 더 필요합니다.' }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $gpu = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null | Select-Object -First 1
        $ErrorActionPreference = $previous
        Note "GPU: $gpu"
    } else {
        Warn 'nvidia-smi 를 찾지 못했습니다. NVIDIA GPU 와 최신 드라이버가 필요합니다. 설치는 계속합니다.'
    }

    Step 2 'uv 준비 (파이썬·패키지 설치 도구)'
    if (-not (Test-Path -LiteralPath $Uv)) {
        $zip = Join-Path $Tools 'uv.zip'
        Get-Verified "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" $zip $UvSha256
        Expand-Archive -LiteralPath $zip -DestinationPath (Join-Path $Tools 'uv') -Force
        Remove-Item -LiteralPath $zip -Force
    }
    Note 'uv 준비됨'

    Step 3 'sd-scripts 받기'
    $marker = Join-Path $SdDir '.installed-commit'
    if ((Test-Path -LiteralPath $marker) -and (Get-Content -LiteralPath $marker -Raw).Trim() -eq $SdCommit) {
        Note '이미 받았습니다'
    } else {
        $zip = Join-Path $Tools 'sd-scripts.zip'
        $unpack = Join-Path $Tools 'sd-scripts-unpack'
        Invoke-WebRequest -Uri "https://github.com/kohya-ss/sd-scripts/archive/$SdCommit.zip" -OutFile $zip -UseBasicParsing
        if (Test-Path -LiteralPath $unpack) { Remove-Item -LiteralPath $unpack -Recurse -Force }
        Expand-Archive -LiteralPath $zip -DestinationPath $unpack -Force
        if (Test-Path -LiteralPath $SdDir) { Remove-Item -LiteralPath $SdDir -Recurse -Force }
        Move-Item -LiteralPath (Join-Path $unpack "sd-scripts-$SdCommit") -Destination $SdDir
        Remove-Item -LiteralPath $unpack -Recurse -Force
        Remove-Item -LiteralPath $zip -Force
        [IO.File]::WriteAllText($marker, $SdCommit)
    }

    Step 4 "파이썬 $PythonVersion 준비"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $Tools 'python'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    if (-not (Test-Path -LiteralPath $Py)) {
        Run $Uv @('venv', '--seed', '--python', $PythonVersion, (Join-Path $SdDir 'venv'))
    }
    Note '가상 환경 준비됨'

    Step 5 'PyTorch 설치 (약 3GB)'
    Run $Uv (@('pip', 'install', '--python', $Py, '--index-url', $TorchIndex) + $TorchPackages)

    Step 6 '학습 패키지 설치'
    Run $Uv @('pip', 'install', '--python', $Py, '-r', (Join-Path $Root 'requirements-lock.txt'))
    $wheel = Join-Path $Tools $Fa2File
    Get-Verified $Fa2Url $wheel $Fa2Sha256
    Run $Uv @('pip', 'install', '--python', $Py, '--no-deps', $wheel)
    Remove-Item -LiteralPath $wheel -Force

    Step 7 '설정 파일 만들기와 점검'
    Run $Py @('-X', 'utf8', (Join-Path $PSScriptRoot 'patch_dataset.py'))
    foreach ($name in 'datasets', 'output', 'logs') { New-Item -ItemType Directory -Force -Path (Join-Path $Root $name) | Out-Null }
    $toolPath = $Root.Replace('\', '/')
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    foreach ($name in 'moss_lora_32gb.toml', 'dataset.toml', 'prompts.txt') {
        $target = Join-Path $Root $name
        if (Test-Path -LiteralPath $target) { Note "$name 은 이미 있어 그대로 둡니다"; continue }
        $text = [IO.File]::ReadAllText((Join-Path $Root "templates\$name"), $utf8).Replace('{{TOOL}}', $toolPath)
        [IO.File]::WriteAllText($target, $text, $utf8)
        Note "$name 만듦"
    }
    Push-Location -LiteralPath $SdDir
    try {
        $env:PYTHONUTF8 = '1'
        Run $Py @('-c', "import torch, flash_attn, triton, moss_image_train_network; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available())")
    } finally { Pop-Location }

    Write-Host "`n설치가 끝났습니다." -ForegroundColor Green
    Write-Host '다음 순서: 모델 파일을 받고(README 참고) run.bat 을 실행하세요.'
    exit 0
} catch {
    Write-Host "`n설치 실패: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host '인터넷 연결과 디스크 공간을 확인한 뒤 install.bat 을 다시 실행하세요. 끝난 단계는 건너뜁니다.'
    exit 1
}
