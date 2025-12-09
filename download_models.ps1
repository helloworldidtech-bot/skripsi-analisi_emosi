$models = @(
    @{ name = 'best_model_70_30.pth'; url = 'PUT_MODEL_URL_HERE' },
    @{ name = 'best_model_80_20.pth'; url = 'PUT_MODEL_URL_HERE' }
)

foreach ($m in $models) {
    $file = $m.name
    $url = $m.url
    if (Test-Path $file) {
        Write-Host "$file already exists — skipping."
        continue
    }
    if ($url -eq 'PUT_MODEL_URL_HERE') {
        Write-Host "Please set the download URL for $file at the top of this script." -ForegroundColor Yellow
        continue
    }
    Write-Host "Downloading $file from $url..."
    Invoke-WebRequest -Uri $url -OutFile $file
    Write-Host "Saved $file"
}
