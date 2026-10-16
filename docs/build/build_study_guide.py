"""Turn docs/study_guide.md into a PDF.

Usage: python docs/build/build_study_guide.py        -> docs/EnduroSense_Study_Guide.pdf

The Markdown is laid out as a Word document (headings, lists, tables, quotes, diagrams) and
Word exports the PDF, so Microsoft Word on Windows is needed. Edit the .md file, not the PDF.
A line `![caption](name)` places the diagram of that name from study_figures.py.
"""
import importlib.util
import re
import subprocess
import sys

from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from endurosense.config import ROOT

BUILD = ROOT / "docs" / "build"
SOURCE = ROOT / "docs" / "study_guide.md"
OUT = ROOT / "docs" / "EnduroSense_Study_Guide.pdf"
WIDTH_CM = 16.4
FIGURE_CM = 16.4
INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*\s][^*]*\*)")
IMAGE = re.compile(r"^!\[(.+)\]\((\w+)\)$")


def load(name):
    spec = importlib.util.spec_from_file_location(name, BUILD / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_inline(par, text: str, size=None, bold=False, italic=False, color=None) -> None:
    """Write text with **bold**, *italic* and `code` spans into a paragraph."""
    for piece in INLINE.split(text):
        if not piece:
            continue
        b, i, mono = bold, italic, False
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            piece, b = piece[2:-2], True
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            piece, mono = piece[1:-1], True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            piece, i = piece[1:-1], True
        if b and "*" in piece:                                   # italic inside bold
            add_inline(par, piece, size, True, italic, color)
            continue
        run = par.add_run(piece)
        run.bold, run.italic = b, i
        if mono:
            run.font.name, run.font.size = "Consolas", Pt((size or 11) - 1)
        elif size:
            run.font.size = Pt(size)
        if color is not None:
            run.font.color.rgb = color


def plain(text: str) -> str:
    return re.sub(r"[*`]", "", text)


def add_page_number(section, color) -> None:
    par = section.footer.paragraphs[0]
    par.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = par.add_run("EnduroSense study guide   |   ")
    run.font.size, run.font.color.rgb = Pt(9), color
    run = par.add_run()
    run.font.size, run.font.color.rgb = Pt(9), color
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = text
        run._r.append(el)


def add_table(doc, rep, lines: list[str]) -> None:
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    header, body = rows[0], [r for r in rows[2:]]
    n = len(header)
    weights = [min(max(max(len(plain(r[j])) for r in [header] + body if j < len(r)), 11), 60) for j in range(n)]
    widths = [WIDTH_CM * w / sum(weights) for w in weights]
    narrow = [w < 2.4 for w in widths]                            # no column so narrow that a short word wraps
    if any(narrow) and not all(narrow):
        spare = WIDTH_CM - 2.4 * sum(narrow)
        rest = sum(w for w, k in zip(widths, narrow) if not k)
        widths = [2.4 if k else w * spare / rest for w, k in zip(widths, narrow)]
    t = doc.add_table(rows=1, cols=n)
    t.style, t.alignment, t.autofit = "Table Grid", WD_TABLE_ALIGNMENT.CENTER, False
    for j, text in enumerate(header):
        cell = t.rows[0].cells[j]
        cell.width = Cm(widths[j])
        rep.set_cell_shade(cell, "DCE8F6")
        add_inline(cell.paragraphs[0], text, size=9.5, bold=True)
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
    for r in body:
        cells = t.add_row().cells
        for j in range(n):
            cells[j].width = Cm(widths[j])
            par = cells[j].paragraphs[0]
            par.paragraph_format.space_after = Pt(2)
            add_inline(par, r[j] if j < len(r) else "", size=9.5)
    for row in t.rows[:-1] if len(t.rows) <= 12 else []:          # keep short tables on one page
        for cell in row.cells:
            cell.paragraphs[0].paragraph_format.keep_with_next = True
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def build_docx(path) -> None:
    rep = load("build_report")
    figs = load("study_figures").figures()
    r = rep.Report()
    doc = r.doc
    add_page_number(doc.sections[0], rep.MUTED)
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if not s or s == "---":
            i += 1
        elif s.startswith("|"):
            j = i
            while j < len(lines) and lines[j].strip().startswith("|"):
                j += 1
            add_table(doc, rep, lines[i:j])
            i = j
        elif IMAGE.match(s):
            caption, name = IMAGE.match(s).groups()
            r.figure(figs[name], caption, width_cm=FIGURE_CM)
            i += 1
        elif s.startswith("# "):
            title = s[2:]
            if title.startswith("Part "):
                doc.add_page_break()
                r.h(title, 1)
            else:                                                 # the document title
                par = r.p(title, size=28, color=rep.BLUE, after=6)
                par.runs[0].bold = True
            i += 1
        elif s.startswith("### "):
            r.h(plain(s[4:]), 3)
            i += 1
        elif s.startswith("## "):
            r.h(plain(s[3:]), 2)
            i += 1
        elif s.startswith(">"):
            text = s[1:].strip()
            if text:
                par = doc.add_paragraph()
                par.paragraph_format.left_indent, par.paragraph_format.space_after = Cm(0.9), Pt(5)
                add_inline(par, text, italic=True, color=rep.MUTED)
            i += 1
        elif re.match(r"^\s*- ", ln):
            level = (len(ln) - len(ln.lstrip())) // 2
            par = doc.add_paragraph(style="List Bullet" if level == 0 else "List Bullet 2")
            par.paragraph_format.space_after = Pt(3)
            add_inline(par, ln.lstrip()[2:])
            i += 1
        elif re.match(r"^\d+\. ", s):                             # the number is written as text, so every list starts at 1
            number, text = s.split(". ", 1)
            par = doc.add_paragraph()
            fmt = par.paragraph_format
            fmt.left_indent, fmt.first_line_indent, fmt.space_after = Cm(1.27), Cm(-0.63), Pt(3)
            fmt.tab_stops.add_tab_stop(Cm(1.27))
            add_inline(par, f"{number}.	{text}")
            i += 1
        else:
            par = doc.add_paragraph()
            add_inline(par, s)
            if s.startswith("**Q. "):                             # a panel question: keep it with its answer
                par.paragraph_format.space_before, par.paragraph_format.keep_with_next = Pt(10), True
            i += 1
    doc.core_properties.title = "EnduroSense study guide"
    doc.core_properties.author = "EnduroSense project"
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def main() -> None:
    docx = BUILD / "_preview" / "study_guide.docx"
    build_docx(docx)
    script = ("$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
              f"$d = $w.Documents.Open('{docx}', $false, $true); $d.SaveAs([ref]'{OUT}', [ref]17); "
              "$pages = $d.ComputeStatistics(2); $d.Close($false); $w.Quit(); \"pages: $pages\"")
    res = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    if res.returncode != 0 or not OUT.exists():
        sys.exit(res.stderr or "Word could not export the PDF")
    print(f"wrote {OUT} ({res.stdout.strip()})")


if __name__ == "__main__":
    main()
