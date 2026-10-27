"""Copy the files needed to regenerate the submission into a separate folder, as the content of the PUBLIC repository
(nothing is pushed or published by this script).

What goes in: the source modules listed in tools/public/files.txt (the modules a from-scratch run of
src/make_submission.py imports, plus the open-data download scripts), requirements.txt, the technical document,
tools/public/README.md as README.md, and a .gitignore. What never goes in: data/, cache/, submissions/, external/,
external_data/, the research log docs/RAPOR.md and reports/ (aggregate numbers of the match).

    python tools/export_public.py <empty target folder>
"""
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITIGNORE = """# match data and anything derived from it -- never commit (terms of use)
data/
cache/
submissions/
external/
external_data/
.venv/
__pycache__/
*.pkl
*.npz
*.npy
"""


def main(dst):
    if os.path.exists(dst) and os.listdir(dst):
        sys.exit("target %s is not empty" % dst)
    with open(os.path.join(ROOT, "tools", "public", "files.txt"), encoding="utf-8") as fh:
        files = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    banned = ("data/", "cache/", "submissions/", "external/", "external_data/", "reports/", "docs/RAPOR.md")
    copies = [(f, f) for f in files] + [("requirements.txt", "requirements.txt"),
                                         ("docs/TECHNICAL_DOCUMENT.md", "docs/TECHNICAL_DOCUMENT.md"),
                                         ("tools/public/README.md", "README.md")]
    for src, out in copies:
        if src.startswith(banned):
            sys.exit("refusing to export %s" % src)
        os.makedirs(os.path.dirname(os.path.join(dst, out)) or dst, exist_ok=True)
        shutil.copy2(os.path.join(ROOT, src), os.path.join(dst, out))
    with open(os.path.join(dst, ".gitignore"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(GITIGNORE)
    print("exported %d files to %s" % (len(copies) + 1, dst))


if __name__ == "__main__":
    main(sys.argv[1])
