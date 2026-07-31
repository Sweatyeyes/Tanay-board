# scripts/capture.ps1
# Lives in the repo (branch main). The laptop pulls a fresh copy before each run,
# so this file can be edited directly on GitHub.
# ASCII only - no Cyrillic here, PowerShell 2.0 on Win7 misreads it.

$DataPath  = "C:\Tanay-data"
$ImageName = "board.png"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)

$bmp.Save((Join-Path $DataPath $ImageName), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()

(Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") | Out-File -FilePath (Join-Path $DataPath "last_update.txt") -Encoding ascii

Set-Location $DataPath
cmd /c "git add -A" 2>&1 | Out-Null
cmd /c "git commit --amend -m board --allow-empty" 2>&1 | Out-Null
cmd /c "git push --force origin data" 2>&1 | Out-Null
