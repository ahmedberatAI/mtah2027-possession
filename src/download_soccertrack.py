"""SoccerTrack v2 (huggingface.co/datasets/atomscott/soccertrack-v2; CC BY 4.0, auto-gated: accept its conditions on the
dataset page first) -> external_data/soccertrack_v2/, resumable, each file checked against the repository's size and
LFS sha256. The access token is read from HF_TOKEN or from ~/.cache/huggingface/token and never printed.

Folders (sizes at revision 325bd4c57e): bas 5 MB (team-labelled ball actions), ball 3 MB (auxiliary ball track,
reconstructed from the annotated events -- not a measurement), raw 2.6 GB (provider tracking exports + calibration),
mot 3.6 GB, gsr 56 GB, videos 83 GB (panoramic halves).

  .venv/Scripts/python.exe -I src/download_soccertrack.py [FOLDER ...]      default: README bas ball raw
"""
import hashlib
import os
import sys
from pathlib import Path

import requests

REPO = "atomscott/soccertrack-v2"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external_data" / "soccertrack_v2"
API = "https://huggingface.co/api/datasets/%s" % REPO


def token():
    t = os.environ.get("HF_TOKEN")
    p = Path.home() / ".cache" / "huggingface" / "token"
    if not t and p.is_file():
        t = p.read_text(encoding="utf-8").strip()
    if not t:
        sys.exit("no Hugging Face token: set HF_TOKEN or save one to %s" % p)
    return t


def listing(session, sha, folder):
    files, url = [], "%s/tree/%s/%s?recursive=true" % (API, sha, folder)
    while url:
        r = session.get(url, timeout=60)
        r.raise_for_status()
        files += [f for f in r.json() if f.get("type") == "file"]
        url = r.links.get("next", {}).get("url")
    return files


def fetch(session, sha, f):
    dest = OUT / f["path"]
    size = f["lfs"]["size"] if f.get("lfs") else f["size"]
    want = f["lfs"].get("oid") if f.get("lfs") else None
    if dest.is_file() and dest.stat().st_size == size:
        return size, False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.is_file() else 0
    headers = {"Range": "bytes=%d-" % have} if have else {}
    with session.get("https://huggingface.co/datasets/%s/resolve/%s/%s" % (REPO, sha, f["path"]), headers=headers,
                     stream=True, timeout=120) as r:
        if r.status_code in (401, 403):
            sys.exit("access refused (%d): accept the dataset's conditions on its page with the token's account"
                     % r.status_code)
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(part, mode) as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    if part.stat().st_size != size:
        sys.exit("size mismatch for %s" % f["path"])
    if want:
        h = hashlib.sha256()
        with open(part, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            part.unlink()
            sys.exit("sha256 mismatch for %s (removed, run again)" % f["path"])
    part.replace(dest)
    return size, True


def main(folders):
    s = requests.Session()
    s.headers["Authorization"] = "Bearer " + token()
    meta = s.get(API, timeout=60)
    meta.raise_for_status()
    sha = meta.json()["sha"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REVISION").write_text(sha + "\n", encoding="utf-8")
    files = [f for f in s.get("%s/tree/%s" % (API, sha), timeout=60).json()
             if f.get("type") == "file" and f["path"].endswith(".md")] if "README" in folders else []
    for folder in folders:
        if folder != "README":
            files += listing(s, sha, folder)
    total = sum(f["lfs"]["size"] if f.get("lfs") else f["size"] for f in files)
    print("SoccerTrack v2 @ %s: %d files, %.2f GB -> %s" % (sha[:10], len(files), total / 1e9, OUT), flush=True)
    done = new = 0
    for i, f in enumerate(files, 1):
        size, fresh = fetch(s, sha, f)
        done += size
        new += fresh
        if fresh or i == len(files):
            print("  %3d/%d files, %.2f / %.2f GB" % (i, len(files), done / 1e9, total / 1e9), flush=True)
    print("done: %d new files" % new)


if __name__ == "__main__":
    main(sys.argv[1:] or ["README", "bas", "ball", "raw"])
