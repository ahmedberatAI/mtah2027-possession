"""Append to docs/RAPOR.md: a note at the end of section 4's latest rounds, a row at the end of section 3's main table, a line in section 1's list of what did not work. Each call is guarded by a unique marker, so running twice cannot duplicate text."""
import io
import os

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "RAPOR.md")
A3 = "### Ekip arkadaşının GitHub kayıtları — 27 Ekim birleştirmesi"
A4 = "**Tur 77 ara sonuç ve tur 78 ön kaydı"
A1 = "**Dersler:**"


def _edit(fn):
    with io.open(PATH, encoding="utf-8", newline="") as fh:
        txt = fh.read()
    nl = "\r\n" if "\r\n" in txt else "\n"
    out = fn(txt, nl)
    with io.open(PATH, "w", encoding="utf-8", newline="") as fh:
        fh.write(out)


def note(marker, text):
    """A paragraph before the anchor in section 4 (the newest notes sit there)."""
    def fn(txt, nl):
        assert marker not in txt, "already written: " + marker
        assert txt.count(A4) == 1
        return txt.replace(A4, text.replace("\n", nl) + nl + nl + A4)
    _edit(fn)
    print("note ok")


def row(marker, text):
    """A row at the end of section 3's main table."""
    def fn(txt, nl):
        assert marker not in txt, "already written: " + marker
        a = nl + nl + A3
        assert txt.count(a) == 1
        return txt.replace(a, nl + text + a)
    _edit(fn)
    print("row ok")


def lost(marker, text):
    """A bullet at the end of section 1's list of what did not work."""
    def fn(txt, nl):
        assert marker not in txt, "already written: " + marker
        a = nl + nl + A1 + nl
        assert txt.count(a) == 1
        return txt.replace(a, nl + "- " + text + a)
    _edit(fn)
    print("lost ok")
