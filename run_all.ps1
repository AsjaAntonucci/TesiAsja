[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
chcp 65001 | Out-Null

$patients    = 1..21 | ForEach-Object { $_.ToString("D2") }
$projectPath = $PWD.Path

foreach ($p in $patients) {
    Write-Host "========================================"
    Write-Host "Paziente chb$p — lancio 3 modelli in parallelo"
    Write-Host "========================================"

    $j1 = Start-Job -ScriptBlock {
        Set-Location $using:projectPath
        $env:PYTHONIOENCODING = "utf-8"
        python src/training/train_ml.py -p $using:p 2>&1
    }
    $j2 = Start-Job -ScriptBlock {
        Set-Location $using:projectPath
        $env:PYTHONIOENCODING = "utf-8"
        python src/training/train_spdnet.py -p $using:p 2>&1
    }
    $j3 = Start-Job -ScriptBlock {
        Set-Location $using:projectPath
        $env:PYTHONIOENCODING = "utf-8"
        python src/training/train_cnn.py -p $using:p 2>&1
    }

    Wait-Job $j1, $j2, $j3 | Out-Null
    Write-Host "--- ML/SVM ---";  Receive-Job $j1
    Write-Host "--- SPDNet ---";  Receive-Job $j2
    Write-Host "--- CNN ---";     Receive-Job $j3
    Remove-Job $j1, $j2, $j3
    Write-Host "chb$p completato`n"
}

Write-Host "Benchmark completo — merge risultati..."
python utils/merge_results.py
Write-Host "Tutto fatto"