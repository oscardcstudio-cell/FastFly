# Garde la mouche en vie sur le port 8010 : la relance si elle tombe OU se fige.
#
# Pourquoi : le 2026-09-19 la mouche est morte d'un MemoryError (PC a 1,4 Go libres sur 16,
# Arena en prend 3,6) et rien ne la relancait ; Oscar l'a vue « hors ligne » en plein set.
# Meme gardien que scripts\beatgrid\serveur.ps1 cote vj-rien.
#
#   powershell -ExecutionPolicy Bypass -File mouche.ps1
#
# Journal : mouche.log. Pour l'arreter : fermer ce script PUIS le python qui ecoute sur 8010.

Set-Location $PSScriptRoot
$log = Join-Path $PSScriptRoot "mouche.log"
$muet = 0

while ($true) {
    $ecoute = Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue
    if (-not $ecoute) {
        Add-Content $log "$(Get-Date -Format s) demarrage"
        Start-Process cmd -WindowStyle Hidden -ArgumentList '/c', ".venv\Scripts\python -u app_server.py --data flywire_v783.bin --port 8010 --audio --beatgrid ../vj-rien >> `"$log`" 2>&1"
        Start-Sleep -Seconds 90  # le connectome met ~30-60 s a charger : ne pas en lancer un deuxieme
        $muet = 0
    } else {
        try { Invoke-WebRequest "http://127.0.0.1:8010/api/params" -TimeoutSec 5 -UseBasicParsing | Out-Null; $muet = 0 }
        catch { $muet++ }
        if ($muet -ge 3) {
            Add-Content $log "$(Get-Date -Format s) figee (3 /api/params sans reponse), arret force puis relance"
            $ecoute | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
            $muet = 0
        }
    }
    Start-Sleep -Seconds 5
}
