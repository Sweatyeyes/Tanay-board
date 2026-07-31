# scripts/capture.ps1
# Живёт в репозитории (ветка main). Ноутбук сам скачивает свежую версию
# перед каждым запуском - править можно прямо на сайте GitHub.

$DataPath  = "C:\Tanay-data"
$ImageName = "board.png"

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

# ---- захват экрана ----
$bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $bounds.Width, $bounds.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)

$bmp.Save((Join-Path $DataPath $ImageName), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose()
$bmp.Dispose()

# ---- отметка времени (по ней страница понимает, жив ли ноутбук) ----
(Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ") |
    Out-File -FilePath (Join-Path $DataPath "last_update.txt") -Encoding ascii -NoNewline

# ---- отправка: всегда переписываем один и тот же коммит ----
Set-Location $DataPath
git add -A
git commit --amend -m "board" --allow-empty 2>&1 | Out-Null
git push --force origin data 2>&1 | Out-Null
