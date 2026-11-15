# Khởi động Label Studio cục bộ (dữ liệu lưu trong data/labeling/ls_data).
# Dùng:  powershell -ExecutionPolicy Bypass -File vision\labeling\run_label_studio.ps1 [-Port 8080]
#
# Môi trường Python của Label Studio nằm NGOÀI repo (mặc định %USERPROFILE%\.venvs\labelstudio) vì
# đường dẫn repo dài + giới hạn MAX_PATH=260 của Windows làm hỏng một số file migration khi import.
# Có thể đổi bằng biến môi trường LS_VENV.
param([int]$Port = 8080)

$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$venv = if ($env:LS_VENV) { $env:LS_VENV } else { Join-Path $env:USERPROFILE ".venvs\labelstudio" }
$exe = Join-Path $venv "Scripts\label-studio.exe"
if (-not (Test-Path $exe)) {
    Write-Error "Chưa cài Label Studio tại $venv. Xem docs/labeling_guideline.md mục 2."
    exit 1
}

$env:LABEL_STUDIO_BASE_DATA_DIR = Join-Path $root "data\labeling\ls_data"
$env:PYTHONUTF8 = "1"
New-Item -ItemType Directory -Force -Path $env:LABEL_STUDIO_BASE_DATA_DIR | Out-Null

Write-Host "Label Studio: http://localhost:$Port  (Ctrl+C để dừng)"
& $exe start --port $Port
