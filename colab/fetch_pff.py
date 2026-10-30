"""Fetch the PFF FC FIFA World Cup 2022 data (Gradient Sports) ON the remote (Colab) VM, from the provider's public
Google Drive folder, into the layout src/gradient.py reads: external_data/gradient/{events,metadata,rosters,tracking}.
Only the 64 match files of each kind are fetched (~3.9 GB in all); the VM's network is used, not the local one.
Access terms: the provider's free access form; the user confirmed the permission on 2026-10-30.

  python colab/fetch_pff.py <repo root>
"""
import os
import re
import sys
import urllib.request

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
FOLDERS = {"events": ("1A7ZvTNzVZ715PDATIhaGA9fPb1KfFmPx", ".json"),
           "metadata": ("1vTCH5W9RMlfWmucTHbrPJ1LrkKnpTJw0", ".json"),
           "rosters": ("1Rp8tpK-MIJt3oSMjufgX6swa9VZZNRy7", ".json"),
           "tracking": ("1-ug2vouBfYazPy8z5lZlgd9abAAJSA5q", ".jsonl.bz2")}
UA = {"User-Agent": "Mozilla/5.0"}
ITEM = re.compile(r'href="https://drive\.google\.com/file/d/([\w-]+)/view[^"]*".*?class="flip-entry-title">([^<]+)<',
                  re.S)


def listing(folder):
    url = "https://drive.google.com/embeddedfolderview?id=%s" % folder
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA)) as r:
        html = r.read().decode("utf-8", "replace")
    return [(fid, name.strip()) for fid, name in ITEM.findall(html)]


def main():
    import gdown
    total = 0
    for kind, (folder, ext) in FOLDERS.items():
        items = [(fid, name) for fid, name in listing(folder) if re.fullmatch(r"\d+" + re.escape(ext), name)]
        dest = os.path.join(ROOT, "external_data", "gradient", kind)
        os.makedirs(dest, exist_ok=True)
        for fid, name in items:
            path = os.path.join(dest, name)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                continue
            out = gdown.download(id=fid, output=path + ".part", quiet=True)
            if not out or not os.path.isfile(path + ".part") or os.path.getsize(path + ".part") == 0:
                raise RuntimeError("download failed: %s/%s" % (kind, name))
            os.replace(path + ".part", path)
        size = sum(os.path.getsize(os.path.join(dest, f)) for f in os.listdir(dest))
        total += size
        print("pff %-8s %d files, %.0f MB" % (kind, len(items), size / 1e6), flush=True)
        if len(items) != 64:
            raise RuntimeError("expected 64 %s files, found %d" % (kind, len(items)))
    print("pff total %.2f GB" % (total / 1e9), flush=True)


if __name__ == "__main__":
    main()
