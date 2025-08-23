# TechnoColabs Delivery AI - Vercel Deployment Script
# PowerShell script for deploying to Vercel

Write-Host "🚀 TechnoColabs Delivery AI - Vercel Deployment" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Check if Vercel CLI is installed
try {
    $vercelVersion = vercel --version
    Write-Host "✅ Vercel CLI found: $vercelVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Vercel CLI not found. Please install it first:" -ForegroundColor Red
    Write-Host "npm install -g vercel" -ForegroundColor Yellow
    exit 1
}

# Check if user is logged in
try {
    vercel whoami | Out-Null
    Write-Host "✅ Logged in to Vercel" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Not logged in to Vercel. Please login first:" -ForegroundColor Yellow
    Write-Host "vercel login" -ForegroundColor Yellow
    exit 1
}

# Check if required files exist
$requiredFiles = @(
    "api/index.py",
    "vercel.json",
    "requirements.txt",
    "best_model.pkl",
    "scaler.pkl",
    "feature_columns.pkl",
    "label_encoders.pkl"
)

Write-Host "📁 Checking required files..." -ForegroundColor Blue
foreach ($file in $requiredFiles) {
    if (Test-Path $file) {
        Write-Host "✅ $file" -ForegroundColor Green
    } else {
        Write-Host "❌ $file (missing)" -ForegroundColor Red
        exit 1
    }
}

# Deploy to Vercel
Write-Host "🚀 Deploying to Vercel..." -ForegroundColor Blue
Write-Host "This may take a few minutes..." -ForegroundColor Yellow

try {
    vercel --prod
    Write-Host "✅ Deployment successful!" -ForegroundColor Green
    Write-Host "🌐 Your app is now live on Vercel!" -ForegroundColor Green
} catch {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
    exit 1
}

Write-Host "🎉 TechnoColabs Delivery AI is now deployed!" -ForegroundColor Green
Write-Host "📱 You can access your app at the URL provided above." -ForegroundColor Blue
