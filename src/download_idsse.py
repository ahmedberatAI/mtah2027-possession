"""Download the open DFL / IDSSE dataset (Bassek et al., CC BY 4.0, figshare 28196177) into
external_data/idsse. Skips files that are already complete. ~2.6 GB."""
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "external_data", "idsse")
API = "https://api.figshare.com/v2/articles/28196177"
UA = {"User-Agent": "Mozilla/5.0"}

if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(API, headers=UA)) as r:
        files = json.load(r)["files"]
    for f in files:
        path = os.path.join(DST, f["name"])
        if os.path.isfile(path) and os.path.getsize(path) == f["size"]:
            continue
        print("downloading", f["name"], "%.0f MB" % (f["size"] / 1e6))
        req = urllib.request.Request(f["download_url"], headers=UA)
        with urllib.request.urlopen(req) as r, open(path, "wb") as out:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    print("done")
