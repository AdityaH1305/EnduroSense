# Assembles the presentation from the JSON written by build_slides.py, by driving Microsoft PowerPoint.
# Usage (normally called by build_slides.py):  powershell -File build_slides.ps1 <spec.json>
param([Parameter(Mandatory = $true)][string]$SpecPath)
$ErrorActionPreference = "Stop"

$spec = Get-Content -Raw -Encoding UTF8 $SpecPath | ConvertFrom-Json
function Rgb([string]$hex) {   # PowerPoint colours are blue-green-red integers
    $r = [Convert]::ToInt32($hex.Substring(0, 2), 16); $g = [Convert]::ToInt32($hex.Substring(2, 2), 16); $b = [Convert]::ToInt32($hex.Substring(4, 2), 16)
    return $r + 256 * $g + 65536 * $b
}
function Pt([double]$inches) { return [single]($inches * 72) }
$alignCode = @{ left = 1; center = 2; right = 3 }

$ppt = New-Object -ComObject PowerPoint.Application
$pres = $ppt.Presentations.Add(0)                      # no window
try {
    $pres.PageSetup.SlideWidth = Pt $spec.width_in
    $pres.PageSetup.SlideHeight = Pt $spec.height_in
    $n = 0
    foreach ($s in $spec.slides) {
        $n++
        $slide = $pres.Slides.Add($n, 12)              # blank layout
        if ($s.title) {
            $t = $slide.Shapes.AddTextbox(1, (Pt $spec.margin), (Pt 0.42), (Pt ($spec.width_in - 2 * $spec.margin)), (Pt 0.9))
            $t.TextFrame.WordWrap = -1
            $r = $t.TextFrame.TextRange
            $r.Text = $s.title; $r.Font.Name = "Calibri"; $r.Font.Size = 30; $r.Font.Bold = -1; $r.Font.Color.RGB = Rgb $spec.title_color
            $line = $slide.Shapes.AddLine((Pt $spec.margin), (Pt 1.3), (Pt ($spec.width_in - $spec.margin)), (Pt 1.3))
            $line.Line.ForeColor.RGB = Rgb "D9D8D3"; $line.Line.Weight = 1
        }
        foreach ($e in $s.elements) {
            switch ($e.type) {
                "text" {
                    $sh = $slide.Shapes.AddTextbox(1, (Pt $e.x), (Pt $e.y), (Pt $e.w), (Pt $e.h))
                    $sh.TextFrame.WordWrap = -1
                    $r = $sh.TextFrame.TextRange
                    $r.Text = $e.text; $r.Font.Name = "Calibri"; $r.Font.Size = $e.size; $r.Font.Color.RGB = Rgb $e.color
                    $r.Font.Bold = $(if ($e.bold) { -1 } else { 0 })
                    $r.ParagraphFormat.Alignment = $alignCode[$e.align]
                    $r.ParagraphFormat.SpaceAfter = $e.after
                    if ($e.bullets) {
                        $r.ParagraphFormat.Bullet.Visible = -1; $r.ParagraphFormat.Bullet.Character = 8226
                        $r.ParagraphFormat.Bullet.Font.Color.RGB = Rgb $spec.title_color
                        $sh.TextFrame.Ruler.Levels.Item(1).FirstMargin = 0; $sh.TextFrame.Ruler.Levels.Item(1).LeftMargin = 20
                    }
                    foreach ($b in $e.bold_ranges) { $r.Characters($b[0], $b[1]).Font.Bold = -1 }
                }
                "image" {
                    $null = $slide.Shapes.AddPicture($e.path, 0, -1, (Pt $e.x), (Pt $e.y), (Pt $e.w), (Pt $e.h))
                }
                "box" {
                    $kind = $(if ($e.shape -eq "arrow") { 33 } else { 5 })     # right arrow / rounded rectangle
                    $sh = $slide.Shapes.AddShape($kind, (Pt $e.x), (Pt $e.y), (Pt $e.w), (Pt $e.h))
                    $sh.Fill.ForeColor.RGB = Rgb $e.fill; $sh.Line.Visible = 0
                    if ($kind -eq 5) { $sh.Adjustments.Item(1) = 0.12 }
                    $r = $sh.TextFrame.TextRange
                    $r.Text = $e.text; $r.Font.Name = "Calibri"; $r.Font.Size = $e.size; $r.Font.Color.RGB = Rgb $e.color
                    $r.Font.Bold = $(if ($e.bold) { -1 } else { 0 })
                    $r.ParagraphFormat.Alignment = 2
                    $sh.TextFrame.WordWrap = -1; $sh.TextFrame.VerticalAnchor = 3
                }
                "table" {
                    $rows = $e.rows.Count; $cols = $e.rows[0].Count
                    $tb = $slide.Shapes.AddTable($rows, $cols, (Pt $e.x), (Pt $e.y), (Pt $e.w), (Pt $e.h)).Table
                    for ($c = 1; $c -le $cols; $c++) { $tb.Columns.Item($c).Width = Pt $e.widths[$c - 1] }
                    for ($i = 1; $i -le $rows; $i++) {
                        for ($c = 1; $c -le $cols; $c++) {
                            $cell = $tb.Cell($i, $c)
                            $r = $cell.Shape.TextFrame.TextRange
                            $r.Text = [string]$e.rows[$i - 1][$c - 1]; $r.Font.Name = "Calibri"; $r.Font.Size = $e.size
                            $r.Font.Color.RGB = Rgb "0B0B0B"; $r.Font.Bold = $(if ($i -eq 1) { -1 } else { 0 })
                            $r.ParagraphFormat.Alignment = $(if ($c -eq 1) { 1 } else { 2 })
                            $cell.Shape.Fill.ForeColor.RGB = Rgb $(if ($i -eq 1) { "DCE8F6" } else { "FFFFFF" })
                        }
                    }
                }
            }
        }
        if ($n -gt 1) {
            $f = $slide.Shapes.AddTextbox(1, (Pt $spec.margin), (Pt ($spec.height_in - 0.45)), (Pt ($spec.width_in - 2 * $spec.margin)), (Pt 0.3))
            $f.TextFrame.TextRange.Text = "$($spec.footer)   |   $n"
            $f.TextFrame.TextRange.Font.Name = "Calibri"; $f.TextFrame.TextRange.Font.Size = 11; $f.TextFrame.TextRange.Font.Color.RGB = Rgb "8A8982"
            $f.TextFrame.TextRange.ParagraphFormat.Alignment = 3
        }
        if ($s.notes) { $slide.NotesPage.Shapes.Placeholders.Item(2).TextFrame.TextRange.Text = $s.notes }
    }
    if (Test-Path $spec.out) { Remove-Item $spec.out -Force }
    $pres.SaveAs($spec.out)
    if ($spec.preview_dir) {
        New-Item -ItemType Directory -Force $spec.preview_dir | Out-Null
        foreach ($slide in $pres.Slides) { $slide.Export((Join-Path $spec.preview_dir ("slide{0:d2}.png" -f $slide.SlideIndex)), "PNG", 1280, 720) }
    }
    "wrote $($spec.out) ($n slides)"
}
finally {
    $pres.Close()
    if ($ppt.Presentations.Count -eq 0) { $ppt.Quit() }
}
