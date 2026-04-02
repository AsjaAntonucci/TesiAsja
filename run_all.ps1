[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

$patients    = 1..21 | ForEach-Object { $_.ToString("D2") }
$projectPath = $PWD.Path

#Write-Host "========================================"
#Write-Host "FASE 1 - PREPROCESSING"
#Write-Host "========================================"

#foreach ($p in $patients) {
  #  Write-Host "Preprocessing chb$p..."
  #  python utils/preprocessing.py -p $p
  #  Write-Host "chb$p preprocessato"
#}

Write-Host "========================================"
Write-Host "FASE 2 - TRAINING"
Write-Host "========================================"

foreach ($p in $patients) {
    Write-Host "========================================"
    Write-Host "Paziente chb$p"
    Write-Host "========================================"

    # CNN da sola (occupa tutta la VRAM)
    Write-Host "--- CNN ---"
    $j1 = Start-Job -ScriptBlock {
        Set-Location $using:projectPath
        $env:PYTHONIOENCODING = "utf-8"
        python src/training/train_cnn.py -p $using:p 2>&1
    }
    Wait-Job $j1 | Out-Null
    Receive-Job $j1
    Remove-Job $j1

    # SPDNet + ML in parallelo (SPDNet GPU, ML solo CPU)
    Write-Host "--- SPDNet + ML ---"
    $j2 = Start-Job -ScriptBlock {
        Set-Location $using:projectPath
        $env:PYTHONIOENCODING = "utf-8"
        python src/training/train_spdnet.py -p $using:p 2>&1
    }
    $j3 = Start-Job -ScriptBlock {
        Set-Location $using:projectPath
        $env:PYTHONIOENCODING = "utf-8"
        python src/training/train_ml.py -p $using:p 2>&1
    }
    Wait-Job $j2, $j3 | Out-Null
    Receive-Job $j2
    Receive-Job $j3
    Remove-Job $j2, $j3

    Write-Host "chb$p completato"
}

Write-Host "Merge risultati..."
python utils/merge_results.py
Write-Host "Tutto fatto!"