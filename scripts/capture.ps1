# scripts/capture.ps1  (temporary version: also collects a window report)
# ASCII only.

$DataPath  = "C:\Tanay-data"
$CodePath  = "C:\Tanay-board"
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

# ---- keep the workflow file inside the data branch ----
# Every push then triggers recognition on Actions. The cron schedule on
# free Actions actually fires only once in a few hours, so this push
# trigger is the reliable path.
try {
  $wfSrc = Join-Path $CodePath "scripts\ocr-data.yml"
  $wfDir = Join-Path $DataPath ".github\workflows"
  if (Test-Path $wfSrc) {
    if (-not (Test-Path $wfDir)) { New-Item -ItemType Directory -Force -Path $wfDir | Out-Null }
    Copy-Item $wfSrc (Join-Path $wfDir "ocr.yml") -Force
  }
} catch { }

# ---- window report ----
try {
  $sig = @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public class WP {
  [DllImport("user32.dll")] public static extern bool EnumWindows(EP cb, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr h, EP cb, IntPtr l);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowTextW(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetClassNameW(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  public delegate bool EP(IntPtr h, IntPtr l);
}
"@
  if (-not ("WP" -as [type])) { Add-Type -TypeDefinition $sig }

  $script:rep = @()
  $script:tops = New-Object System.Collections.ArrayList

  $cb = [WP+EP]{
    param($h, $l)
    if ([WP]::IsWindowVisible($h)) {
      $sb = New-Object System.Text.StringBuilder 512
      [void][WP]::GetWindowTextW($h, $sb, 512)
      $cn = New-Object System.Text.StringBuilder 256
      [void][WP]::GetClassNameW($h, $cn, 256)
      $script:rep += ("WIN class=" + $cn.ToString() + " | title=" + $sb.ToString())
      [void]$script:tops.Add($h)
    }
    return $true
  }
  [void][WP]::EnumWindows($cb, [IntPtr]::Zero)

  $script:rep += ""
  $script:rep += "=== CHILDREN ==="
  foreach ($t in $script:tops) {
    $sb = New-Object System.Text.StringBuilder 512
    [void][WP]::GetWindowTextW($t, $sb, 512)
    $cn = New-Object System.Text.StringBuilder 256
    [void][WP]::GetClassNameW($t, $cn, 256)
    $script:rep += ""
    $script:rep += ("--- " + $cn.ToString() + " | " + $sb.ToString())
    $script:n = 0
    $cbc = [WP+EP]{
      param($h, $l)
      if ($script:n -lt 100) {
        $s1 = New-Object System.Text.StringBuilder 1024
        [void][WP]::GetWindowTextW($h, $s1, 1024)
        $s2 = New-Object System.Text.StringBuilder 256
        [void][WP]::GetClassNameW($h, $s2, 256)
        $script:rep += ("   " + $s2.ToString() + " | " + $s1.ToString())
        $script:n++
      }
      return $true
    }
    [void][WP]::EnumChildWindows($t, $cbc, [IntPtr]::Zero)
    if ($script:n -eq 0) { $script:rep += "   (custom drawn, no controls)" }
  }
  $script:rep | Out-File -FilePath (Join-Path $DataPath "probe_result.txt") -Encoding UTF8
} catch {
  ("probe failed: " + $_) | Out-File -FilePath (Join-Path $DataPath "probe_result.txt") -Encoding UTF8
}

Set-Location $DataPath
cmd /c "git add -A" 2>&1 | Out-Null
cmd /c "git commit --amend -m board --allow-empty" 2>&1 | Out-Null
cmd /c "git push --force origin data" 2>&1 | Out-Null
