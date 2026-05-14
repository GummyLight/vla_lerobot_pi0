param(
    [switch]$CopyDefaultDataset,
    [switch]$CopyAutoConDataset,
    [switch]$CopyHfCache
)

$ErrorActionPreference = "Stop"

$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ParentRoot = Split-Path -Parent $BundleRoot

function Copy-Directory($Source, $Destination) {
    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Host "missing: $Source"
        return
    }

    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Recurse -Force
    Write-Host "copied: $Source -> $Destination"
}

if ($CopyDefaultDataset) {
    Copy-Directory `
        -Source (Join-Path $ParentRoot "datasets/open_3d_printer_diversified") `
        -Destination (Join-Path $BundleRoot "datasets/open_3d_printer_diversified")
}

if ($CopyAutoConDataset) {
    Copy-Directory `
        -Source (Join-Path $ParentRoot "datasets/dataset_AutoCon") `
        -Destination (Join-Path $BundleRoot "datasets/dataset_AutoCon")
}

if ($CopyHfCache) {
    Copy-Directory `
        -Source "D:\.hfcache\hub\models--lerobot--pi0" `
        -Destination (Join-Path $BundleRoot "hf_cache/hub/models--lerobot--pi0")

    Copy-Directory `
        -Source "D:\.hfcache\hub\models--google--paligemma-3b-pt-224" `
        -Destination (Join-Path $BundleRoot "hf_cache/hub/models--google--paligemma-3b-pt-224")
}

if (-not ($CopyDefaultDataset -or $CopyAutoConDataset -or $CopyHfCache)) {
    Write-Host "Nothing selected."
    Write-Host "Examples:"
    Write-Host "  ./prepare_from_parent.ps1 -CopyDefaultDataset"
    Write-Host "  ./prepare_from_parent.ps1 -CopyDefaultDataset -CopyHfCache"
}
