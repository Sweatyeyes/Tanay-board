# scripts/capture.ps1
# ASCII only: file travels through a Windows 7 machine with PowerShell 2.0.
#
# The scheduler still starts this once a minute. Inside, we take several
# screenshots with a short pause and push only when the board actually
# changed - that keeps traffic and Actions runs low while cutting the delay
# from 60 seconds down to SHOT_INTERVAL.
#
# Comparison ignores the top strip with the clock: it ticks every minute
# and would make every frame look "changed".

$DataPath      = "C:\Tanay-data"
$CodePath      = "C:\Tanay-board"
$ImageName     = "board.png"
$SHOTS         = 4      # snapshots per launch
$SHOT_INTERVAL = 15     # seconds between snapshots
$HEARTBEAT_SEC = 150    # push at least this often even without changes
$SKIP_TOP_PX   = 40     # ignore the clock strip when comparing

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# ---- keep the workflow file inside the data branch (push-triggered OCR) ----
try {
  $wfSrc = Join-Path $CodePath "scripts\ocr-data.yml"
  $wfDir = Join-Path $DataPath ".github\workflows"
  if (Test-Path $wfSrc) {
    if (-not (Test-Path $wfDir)) { New-Item -ItemType Directory -Force -Path $wfDir | Out-Null }
    Copy-Item $wfSrc (Join-Path $wfDir "ocr.yml") -Force
  }
} catch { }

function Get-BoardHash($bmp) {
  # MD5 over the image below the clock strip; PowerShell 2.0 has no Get-FileHash
  $h = $bmp.Height - $SKIP_TOP_PX
  if ($h -lt 10) { $h = $bmp.Height }
  $rect = New-Object System.Drawing.Rectangle 0, $SKIP_TOP_PX, $bmp.Width, $h
  $part = $bmp.Clone($rect, $bmp.PixelFormat)
  $ms = New-Object System.IO.MemoryStream
  $part.Save($ms, [System.Drawing.Imaging.ImageFormat]::Bmp)
  $bytes = $ms.ToArray()
  $ms.Close()
  $part.Dispose()
  $md5 = [System.Security.Cryptography.MD5]::Create()
  $sum = $md5.ComputeHash($bytes)
  return [System.BitConverter]::ToString($sum)
}

function Push-Board {
  Set-Location $DataPath
  cmd /c "git add -A" 2>&1 | Out-Null
  cmd /c "git commit --amend -m board --allow-empty" 2>&1 | Out-Null
  cmd /c "git push --force origin data" 2>&1 | Out-Null
}

$stateFile = Join-Path $DataPath "last_hash.txt"
$lastHash = ""
if (Test-Path $stateFile) { $lastHash = (Get-Content $stateFile -ErrorAction SilentlyContinue | Select-Object -First 1) }

$lastPush = [DateTime]::MinValue
$pushFile = Join-Path $DataPath "last_update.txt"
if (Test-Path $pushFile) {
  try { $lastPush = [DateTime]::Parse((Get-Content $pushFile | Select-Object -First 1)).ToUniversalTime() } catch { }
}

$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds

for ($i = 0; $i -lt $SHOTS; $i++) {
  $started = Get-Date

  $bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
  $g = [System.Drawing.Graphics]::FromImage($bmp)
  $g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)

  $hash = ""
  try { $hash = Get-BoardHash $bmp } catch { $hash = "" }

  $age = ([DateTime]::UtcNow - $lastPush).TotalSeconds
  $changed = ($hash -eq "") -or ($hash -ne $lastHash)

  if ($changed -or ($age -ge $HEARTBEAT_SEC)) {
    $bmp.Save((Join-Path $DataPath $ImageName), [System.Drawing.Imaging.ImageFormat]::Png)
    ([DateTime]::UtcNow).ToString("yyyy-MM-ddTHH:mm:ssZ") | Out-File -FilePath $pushFile -Encoding ascii
    $hash | Out-File -FilePath $stateFile -Encoding ascii
    Push-Board
    $lastHash = $hash
    $lastPush = [DateTime]::UtcNow
  }

  $g.Dispose()
  $bmp.Dispose()

  if ($i -lt ($SHOTS - 1)) {
    $spent = ((Get-Date) - $started).TotalSeconds
    $wait = $SHOT_INTERVAL - $spent
    if ($wait -gt 0) { Start-Sleep -Seconds ([int]$wait) }
  }
}
