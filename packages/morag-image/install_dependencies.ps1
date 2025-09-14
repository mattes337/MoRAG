# PowerShell script for installing morag-image dependencies on Windows

Write-Host "Installing system dependencies for morag-image package..." -ForegroundColor Green

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "This script requires administrator privileges. Please run PowerShell as administrator." -ForegroundColor Red
    exit 1
}

# Check if Chocolatey is installed
if (-not (Get-Command choco -ErrorAction SilentlyContinue)) {
    Write-Host "Chocolatey is not installed. Installing Chocolatey..." -ForegroundColor Yellow
    Set-ExecutionPolicy Bypass -Scope Process -Force
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
    Invoke-Expression ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))
    
    # Refresh environment variables
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# Install Tesseract OCR
Write-Host "Installing Tesseract OCR..." -ForegroundColor Cyan
try {
    choco install tesseract -y
    
    # Add Tesseract to PATH if not already there
    $tesseractPath = "C:\Program Files\Tesseract-OCR"
    if (-not ($env:Path -like "*$tesseractPath*")) {
        [Environment]::SetEnvironmentVariable("Path", $env:Path + ";$tesseractPath", "Machine")
        $env:Path = $env:Path + ";$tesseractPath"
    }
    
    Write-Host "Tesseract OCR installed successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to install Tesseract OCR: $_" -ForegroundColor Red
    Write-Host "Please install Tesseract OCR manually from: https://github.com/UB-Mannheim/tesseract/wiki" -ForegroundColor Yellow
}

# Install Visual C++ Redistributable (required for OpenCV)
Write-Host "Installing Visual C++ Redistributable..." -ForegroundColor Cyan
try {
    choco install vcredist140 -y
    Write-Host "Visual C++ Redistributable installed successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to install Visual C++ Redistributable: $_" -ForegroundColor Red
    Write-Host "Please install Visual C++ Redistributable manually." -ForegroundColor Yellow
}

# Verify Tesseract installation
if (Get-Command tesseract -ErrorAction SilentlyContinue) {
    Write-Host "Tesseract OCR installation verified:" -ForegroundColor Green
    tesseract --version | Select-Object -First 1
} else {
    Write-Host "WARNING: Tesseract OCR installation could not be verified." -ForegroundColor Yellow
    Write-Host "You may need to restart your PowerShell session or check your PATH." -ForegroundColor Yellow
}

# Install Python packages
Write-Host "\nInstalling Python packages..." -ForegroundColor Cyan
try {
    pip install morag-image
    Write-Host "morag-image package installed successfully." -ForegroundColor Green
} catch {
    Write-Host "Failed to install morag-image package: $_" -ForegroundColor Red
    Write-Host "Please install the package manually using: pip install morag-image" -ForegroundColor Yellow
}

Write-Host "\nNext steps:" -ForegroundColor Green
Write-Host "1. Set up your Google API key for Gemini Vision (if using captioning):" -ForegroundColor White
Write-Host "   $env:GOOGLE_API_KEY = 'your-api-key'" -ForegroundColor White
Write-Host "\nFor more information, see the README.md file." -ForegroundColor White